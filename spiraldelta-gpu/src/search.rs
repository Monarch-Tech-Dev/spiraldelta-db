/*!
GPU-Accelerated Search Algorithms for SpiralDelta

High-performance implementations of various similarity search algorithms optimized for GPU execution.
*/

use candle_core::{Device, Tensor};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use std::sync::Arc;
use anyhow::{Result, anyhow};
use crate::memory::GpuMemoryManager;

pub struct GpuSearchEngine {
    device: Arc<CudaDevice>,
    memory_manager: Arc<GpuMemoryManager>,
}

impl GpuSearchEngine {
    pub fn new(device: Arc<CudaDevice>, memory_manager: Arc<GpuMemoryManager>) -> Self {
        Self {
            device,
            memory_manager,
        }
    }
    
    /// Perform batched similarity search with optimal GPU utilization
    pub fn batched_similarity_search(
        &self,
        queries: &Tensor,
        index: &Tensor,
        k: usize,
        metric: SimilarityMetric,
        batch_size: usize,
    ) -> Result<SearchResults> {
        let query_count = queries.shape()[0];
        let index_count = index.shape()[0];
        let dimensions = queries.shape()[1];
        
        // Validate inputs
        if index.shape()[1] != dimensions {
            return Err(anyhow!("Query and index dimensions must match"));
        }
        
        let mut all_indices = Vec::new();
        let mut all_scores = Vec::new();
        
        // Process in batches to manage memory
        for batch_start in (0..query_count).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(query_count);
            let batch_queries = queries.narrow(0, batch_start, batch_end - batch_start)?;
            
            let (batch_indices, batch_scores) = self.search_batch(
                &batch_queries, 
                index, 
                k, 
                metric
            )?;
            
            all_indices.extend(batch_indices);
            all_scores.extend(batch_scores);
        }
        
        Ok(SearchResults {
            indices: all_indices,
            scores: all_scores,
            query_count: query_count,
            k: k,
        })
    }
    
    fn search_batch(
        &self,
        queries: &Tensor,
        index: &Tensor,
        k: usize,
        metric: SimilarityMetric,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
        match metric {
            SimilarityMetric::Cosine => self.cosine_similarity_batch(queries, index, k),
            SimilarityMetric::Euclidean => self.euclidean_distance_batch(queries, index, k),
            SimilarityMetric::DotProduct => self.dot_product_batch(queries, index, k),
            SimilarityMetric::Manhattan => self.manhattan_distance_batch(queries, index, k),
        }
    }
    
    fn cosine_similarity_batch(
        &self,
        queries: &Tensor,
        index: &Tensor,
        k: usize,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
        // Normalize vectors for cosine similarity
        let queries_norm = self.l2_normalize(queries)?;
        let index_norm = self.l2_normalize(index)?;
        
        // Compute similarity matrix using optimized GEMM
        let similarities = self.optimized_matmul(&queries_norm, &index_norm.t()?)?;
        
        // Extract top-k using GPU-optimized selection
        self.extract_topk_gpu(&similarities, k, true)
    }
    
    fn euclidean_distance_batch(
        &self,
        queries: &Tensor,
        index: &Tensor,
        k: usize,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
        // Efficient squared distance computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        let queries_sq = queries.sqr()?.sum_keepdim(1)?;
        let index_sq = index.sqr()?.sum_keepdim(1)?;
        let cross_term = self.optimized_matmul(queries, &index.t()?)?.broadcast_mul(&Tensor::new(-2.0, &queries.device())?)?;
        
        let distances = queries_sq.broadcast_add(&index_sq.t()?)?.add(&cross_term)?;
        
        // For k-nearest, we want smallest distances (so negate for top-k selection)
        let neg_distances = distances.neg()?;
        self.extract_topk_gpu(&neg_distances, k, true)
    }
    
    fn dot_product_batch(
        &self,
        queries: &Tensor,
        index: &Tensor,
        k: usize,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
        let dot_products = self.optimized_matmul(queries, &index.t()?)?;
        self.extract_topk_gpu(&dot_products, k, true)
    }
    
    fn manhattan_distance_batch(
        &self,
        queries: &Tensor,
        index: &Tensor,
        k: usize,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
        let batch_size = queries.shape()[0];
        let index_size = index.shape()[0];
        let dimensions = queries.shape()[1];
        
        // Manhattan distance requires element-wise operations
        // Create expanded tensors for broadcasting
        let queries_expanded = queries.unsqueeze(1)?.broadcast_as((batch_size, index_size, dimensions))?;
        let index_expanded = index.unsqueeze(0)?.broadcast_as((batch_size, index_size, dimensions))?;
        
        // Compute |a - b| and sum along dimension axis
        let differences = queries_expanded.sub(&index_expanded)?.abs()?;
        let distances = differences.sum(2)?;
        
        // Negate for top-k (we want smallest distances)
        let neg_distances = distances.neg()?;
        self.extract_topk_gpu(&neg_distances, k, true)
    }
    
    fn l2_normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        let norms = tensor.sqr()?.sum_keepdim(1)?.sqrt()?;
        let eps = Tensor::new(1e-8f32, &tensor.device())?;
        let safe_norms = norms.add(&eps)?;
        tensor.broadcast_div(&safe_norms)
    }
    
    fn optimized_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Use cuBLAS for optimized matrix multiplication
        // This would use mixed precision if enabled in config
        a.matmul(b)
    }
    
    fn extract_topk_gpu(
        &self,
        similarities: &Tensor,
        k: usize,
        largest: bool,
    ) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
        // For now, use CPU-based top-k extraction
        // In production, this would use GPU kernels for better performance
        let cpu_similarities = similarities.to_device(&Device::Cpu)?;
        let sim_data = cpu_similarities.to_vec2::<f32>()?;
        
        let results: Vec<(Vec<usize>, Vec<f32>)> = sim_data
            .iter()
            .map(|row| {
                let mut indexed: Vec<(usize, f32)> = row
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| (i, val))
                    .collect();
                
                // Sort by score
                if largest {
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }
                
                indexed.truncate(k);
                
                let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
                let scores: Vec<f32> = indexed.iter().map(|(_, s)| *s).collect();
                
                (indices, scores)
            })
            .collect();
        
        let all_indices = results.iter().map(|(indices, _)| indices.clone()).collect();
        let all_scores = results.iter().map(|(_, scores)| scores.clone()).collect();
        
        Ok((all_indices, all_scores))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

pub struct SearchResults {
    pub indices: Vec<Vec<usize>>,
    pub scores: Vec<Vec<f32>>,
    pub query_count: usize,
    pub k: usize,
}

impl SearchResults {
    pub fn get_result(&self, query_idx: usize) -> Option<(&[usize], &[f32])> {
        if query_idx < self.query_count {
            Some((&self.indices[query_idx], &self.scores[query_idx]))
        } else {
            None
        }
    }
    
    pub fn to_python_format(&self) -> Vec<Vec<(usize, f32)>> {
        self.indices
            .iter()
            .zip(self.scores.iter())
            .map(|(indices, scores)| {
                indices
                    .iter()
                    .zip(scores.iter())
                    .map(|(&idx, &score)| (idx, score))
                    .collect()
            })
            .collect()
    }
}

pub struct ApproximateSearchEngine {
    device: Arc<CudaDevice>,
    memory_manager: Arc<GpuMemoryManager>,
}

impl ApproximateSearchEngine {
    pub fn new(device: Arc<CudaDevice>, memory_manager: Arc<GpuMemoryManager>) -> Self {
        Self {
            device,
            memory_manager,
        }
    }
    
    /// Approximate nearest neighbor search using LSH
    pub fn lsh_search(
        &self,
        queries: &Tensor,
        index: &Tensor,
        k: usize,
        num_hashes: usize,
        hash_dim: usize,
    ) -> Result<SearchResults> {
        // Generate random hash functions
        let hash_functions = self.generate_hash_functions(index.shape()[1], hash_dim, num_hashes)?;
        
        // Hash all index vectors
        let index_hashes = self.compute_hashes(index, &hash_functions)?;
        
        // Hash query vectors and find candidates
        let query_hashes = self.compute_hashes(queries, &hash_functions)?;
        
        // Find candidate sets for each query
        let candidates = self.find_candidates(&query_hashes, &index_hashes)?;
        
        // Refine search within candidates
        self.refine_candidates(queries, index, candidates, k)
    }
    
    fn generate_hash_functions(
        &self,
        input_dim: usize,
        hash_dim: usize,
        num_hashes: usize,
    ) -> Result<Vec<Tensor>> {
        let device = &Device::Cuda(self.device.clone());
        
        (0..num_hashes)
            .map(|_| {
                // Generate random Gaussian matrix for LSH
                Tensor::randn(0f32, 1f32, (hash_dim, input_dim), device)
            })
            .collect::<Result<Vec<_>, _>>()
    }
    
    fn compute_hashes(&self, vectors: &Tensor, hash_functions: &[Tensor]) -> Result<Vec<Tensor>> {
        hash_functions
            .iter()
            .map(|hash_fn| {
                let projected = vectors.matmul(hash_fn)?;
                // Sign-based hashing
                let zeros = Tensor::zeros_like(&projected)?;
                projected.ge(&zeros)?.to_dtype(candle_core::DType::F32)
            })
            .collect()
    }
    
    fn find_candidates(
        &self,
        query_hashes: &[Tensor],
        index_hashes: &[Tensor],
    ) -> Result<Vec<Vec<usize>>> {
        // Simplified candidate finding - would be optimized with hash tables
        let query_count = query_hashes[0].shape()[0];
        let index_count = index_hashes[0].shape()[0];
        
        let mut candidates = vec![Vec::new(); query_count];
        
        for query_idx in 0..query_count {
            for index_idx in 0..index_count {
                let mut matches = 0;
                
                // Count hash function matches
                for (query_hash, index_hash) in query_hashes.iter().zip(index_hashes.iter()) {
                    let query_val = query_hash.get(query_idx)?.to_scalar::<f32>()?;
                    let index_val = index_hash.get(index_idx)?.to_scalar::<f32>()?;
                    
                    if (query_val - index_val).abs() < 0.1 {
                        matches += 1;
                    }
                }
                
                // Add to candidates if enough matches
                if matches >= query_hashes.len() / 2 {
                    candidates[query_idx].push(index_idx);
                }
            }
        }
        
        Ok(candidates)
    }
    
    fn refine_candidates(
        &self,
        queries: &Tensor,
        index: &Tensor,
        candidates: Vec<Vec<usize>>,
        k: usize,
    ) -> Result<SearchResults> {
        let mut all_indices = Vec::new();
        let mut all_scores = Vec::new();
        
        for (query_idx, candidate_list) in candidates.iter().enumerate() {
            if candidate_list.is_empty() {
                // No candidates found, return empty result
                all_indices.push(vec![]);
                all_scores.push(vec![]);
                continue;
            }
            
            // Extract query vector
            let query = queries.get(query_idx)?;
            
            // Compute exact similarities for candidates only
            let mut similarities = Vec::new();
            for &candidate_idx in candidate_list {
                let candidate = index.get(candidate_idx)?;
                let similarity = self.compute_cosine_similarity(&query, &candidate)?;
                similarities.push((candidate_idx, similarity));
            }
            
            // Sort and take top-k
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            similarities.truncate(k);
            
            let indices: Vec<usize> = similarities.iter().map(|(idx, _)| *idx).collect();
            let scores: Vec<f32> = similarities.iter().map(|(_, score)| *score).collect();
            
            all_indices.push(indices);
            all_scores.push(scores);
        }
        
        Ok(SearchResults {
            indices: all_indices,
            scores: all_scores,
            query_count: queries.shape()[0],
            k,
        })
    }
    
    fn compute_cosine_similarity(&self, a: &Tensor, b: &Tensor) -> Result<f32> {
        let dot_product = a.mul(b)?.sum_all()?.to_scalar::<f32>()?;
        let norm_a = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let norm_b = b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        
        Ok(dot_product / (norm_a * norm_b + 1e-8))
    }
}