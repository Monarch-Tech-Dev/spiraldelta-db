/*!
GPU-Accelerated Index Construction for SpiralDelta

High-performance index building algorithms optimized for GPU execution.
*/

use candle_core::{Device, Tensor};
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::memory::GpuMemoryManager;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuIndexType {
    HNSW {
        levels: Vec<HNSWLevel>,
        entry_point: usize,
        max_connections: usize,
        ef_construction: usize,
    },
    IVF {
        centroids: Vec<Vec<f32>>,
        inverted_lists: HashMap<usize, Vec<usize>>,
        quantizer: QuantizerConfig,
    },
    Spiral {
        spiral_coords: Vec<(f32, f32)>,
        ordering: Vec<usize>,
        compression_params: CompressionParams,
    },
    LSH {
        hash_functions: Vec<Vec<f32>>,
        hash_tables: Vec<HashMap<u64, Vec<usize>>>,
        num_hashes: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWLevel {
    pub level: usize,
    pub connections: HashMap<usize, Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizerConfig {
    pub quantization_type: QuantizationType,
    pub bits_per_component: u8,
    pub scale_factors: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationType {
    Uniform,
    KMeans,
    ProductQuantization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionParams {
    pub delta_threshold: f32,
    pub compression_ratio: f32,
    pub spiral_density: f32,
}

pub struct GpuIndexBuilder {
    device: Arc<CudaDevice>,
    memory_manager: Arc<GpuMemoryManager>,
}

impl GpuIndexBuilder {
    pub fn new(device: Arc<CudaDevice>, memory_manager: Arc<GpuMemoryManager>) -> Self {
        Self {
            device,
            memory_manager,
        }
    }
    
    /// Build HNSW index with GPU acceleration
    pub fn build_hnsw_index(
        &self,
        vectors: &Tensor,
        max_connections: usize,
        ef_construction: usize,
        max_level: usize,
    ) -> Result<GpuIndexType> {
        let vector_count = vectors.shape()[0];
        let dimensions = vectors.shape()[1];
        
        log::info!("Building HNSW index for {} vectors of {} dimensions", vector_count, dimensions);
        
        // Initialize levels with geometric probability distribution
        let levels = self.generate_hnsw_levels(vector_count, max_level)?;
        
        // Build connections level by level (bottom-up)
        let mut hnsw_levels = Vec::new();
        
        for level in 0..=max_level {
            let level_vectors: Vec<usize> = levels
                .iter()
                .enumerate()
                .filter(|(_, &l)| l >= level)
                .map(|(i, _)| i)
                .collect();
            
            if level_vectors.is_empty() {
                break;
            }
            
            let connections = self.build_hnsw_level(
                vectors,
                &level_vectors,
                max_connections,
                ef_construction,
            )?;
            
            hnsw_levels.push(HNSWLevel {
                level,
                connections,
            });
        }
        
        // Find entry point (highest level vector)
        let entry_point = levels
            .iter()
            .enumerate()
            .max_by_key(|(_, &level)| level)
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        Ok(GpuIndexType::HNSW {
            levels: hnsw_levels,
            entry_point,
            max_connections,
            ef_construction,
        })
    }
    
    /// Build IVF (Inverted File) index with GPU acceleration
    pub fn build_ivf_index(
        &self,
        vectors: &Tensor,
        num_clusters: usize,
        quantization: QuantizerConfig,
    ) -> Result<GpuIndexType> {
        let vector_count = vectors.shape()[0];
        let dimensions = vectors.shape()[1];
        
        log::info!("Building IVF index with {} clusters for {} vectors", num_clusters, vector_count);
        
        // Run K-means clustering on GPU
        let centroids = self.gpu_kmeans_clustering(vectors, num_clusters)?;
        
        // Assign vectors to clusters
        let assignments = self.assign_to_clusters(vectors, &centroids)?;
        
        // Build inverted lists
        let mut inverted_lists = HashMap::new();
        for (vector_idx, cluster_id) in assignments.iter().enumerate() {
            inverted_lists
                .entry(*cluster_id)
                .or_insert_with(Vec::new)
                .push(vector_idx);
        }
        
        // Convert centroids tensor to vectors
        let centroid_vecs = self.tensor_to_vectors(&centroids)?;
        
        Ok(GpuIndexType::IVF {
            centroids: centroid_vecs,
            inverted_lists,
            quantizer: quantization,
        })
    }
    
    /// Build optimized Spiral index with GPU acceleration
    pub fn build_spiral_index(
        &self,
        vectors: &Tensor,
        compression_params: CompressionParams,
    ) -> Result<GpuIndexType> {
        let vector_count = vectors.shape()[0];
        let dimensions = vectors.shape()[1];
        
        log::info!("Building Spiral index for {} vectors with {:.2}% compression target", 
                  vector_count, compression_params.compression_ratio * 100.0);
        
        // Compute spiral coordinates on GPU
        let spiral_coords = self.compute_spiral_coordinates_gpu(vectors, &compression_params)?;
        
        // Generate spiral ordering
        let ordering = self.generate_spiral_ordering(&spiral_coords)?;
        
        Ok(GpuIndexType::Spiral {
            spiral_coords,
            ordering,
            compression_params,
        })
    }
    
    /// Build LSH index with GPU acceleration
    pub fn build_lsh_index(
        &self,
        vectors: &Tensor,
        num_hashes: usize,
        hash_dim: usize,
    ) -> Result<GpuIndexType> {
        let vector_count = vectors.shape()[0];
        let dimensions = vectors.shape()[1];
        
        log::info!("Building LSH index with {} hash functions for {} vectors", num_hashes, vector_count);
        
        // Generate random hash functions
        let hash_functions = self.generate_lsh_functions(dimensions, hash_dim, num_hashes)?;
        
        // Compute hashes for all vectors
        let vector_hashes = self.compute_vector_hashes(vectors, &hash_functions)?;
        
        // Build hash tables
        let hash_tables = self.build_hash_tables(&vector_hashes)?;
        
        Ok(GpuIndexType::LSH {
            hash_functions,
            hash_tables,
            num_hashes,
        })
    }
    
    // Private helper methods
    
    fn generate_hnsw_levels(&self, vector_count: usize, max_level: usize) -> Result<Vec<usize>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        (0..vector_count)
            .map(|_| {
                // Geometric distribution for level assignment
                let mut level = 0;
                while level < max_level && rng.gen::<f32>() < 0.5 {
                    level += 1;
                }
                Ok(level)
            })
            .collect()
    }
    
    fn build_hnsw_level(
        &self,
        vectors: &Tensor,
        level_vectors: &[usize],
        max_connections: usize,
        ef_construction: usize,
    ) -> Result<HashMap<usize, Vec<usize>>> {
        let mut connections = HashMap::new();
        
        for &vector_idx in level_vectors {
            // Find nearest neighbors for this vector
            let query = vectors.get(vector_idx)?;
            let candidates = self.find_construction_candidates(
                &query,
                vectors,
                level_vectors,
                ef_construction,
            )?;
            
            // Select best connections using heuristic
            let selected = self.select_hnsw_connections(
                &query,
                vectors,
                &candidates,
                max_connections,
            )?;
            
            connections.insert(vector_idx, selected);
        }
        
        // Make connections bidirectional
        self.make_bidirectional_connections(&mut connections, max_connections);
        
        Ok(connections)
    }
    
    fn find_construction_candidates(
        &self,
        query: &Tensor,
        vectors: &Tensor,
        candidates: &[usize],
        ef: usize,
    ) -> Result<Vec<usize>> {
        // Simple linear search for construction
        // In practice, would use more sophisticated search
        let mut similarities = Vec::new();
        
        for &candidate_idx in candidates {
            let candidate = vectors.get(candidate_idx)?;
            let similarity = self.compute_cosine_similarity(query, &candidate)?;
            similarities.push((candidate_idx, similarity));
        }
        
        // Sort by similarity and take top ef
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(ef);
        
        Ok(similarities.into_iter().map(|(idx, _)| idx).collect())
    }
    
    fn select_hnsw_connections(
        &self,
        _query: &Tensor,
        _vectors: &Tensor,
        candidates: &[usize],
        max_connections: usize,
    ) -> Result<Vec<usize>> {
        // Simplified connection selection
        // In practice, would use more sophisticated heuristics
        Ok(candidates.iter().take(max_connections).copied().collect())
    }
    
    fn make_bidirectional_connections(
        &self,
        connections: &mut HashMap<usize, Vec<usize>>,
        max_connections: usize,
    ) {
        let mut updates = Vec::new();
        
        for (&from, to_list) in connections.iter() {
            for &to in to_list {
                if !connections.get(&to).unwrap_or(&Vec::new()).contains(&from) {
                    updates.push((to, from));
                }
            }
        }
        
        for (to, from) in updates {
            let entry = connections.entry(to).or_insert_with(Vec::new);
            if entry.len() < max_connections {
                entry.push(from);
            }
        }
    }
    
    fn gpu_kmeans_clustering(&self, vectors: &Tensor, k: usize) -> Result<Tensor> {
        let vector_count = vectors.shape()[0];
        let dimensions = vectors.shape()[1];
        
        // Initialize centroids randomly
        let indices: Vec<usize> = (0..k).map(|i| i * vector_count / k).collect();
        let mut centroids = Tensor::stack(
            &indices.iter().map(|&i| vectors.get(i)).collect::<Result<Vec<_>, _>>()?,
            0
        )?;
        
        // K-means iterations
        for iteration in 0..100 {
            // Assign points to clusters
            let assignments = self.assign_to_clusters(vectors, &centroids)?;
            
            // Update centroids
            let new_centroids = self.update_centroids(vectors, &assignments, k)?;
            
            // Check for convergence
            let diff = centroids.sub(&new_centroids)?.abs()?.sum_all()?.to_scalar::<f32>()?;
            centroids = new_centroids;
            
            if diff < 1e-6 {
                log::debug!("K-means converged after {} iterations", iteration + 1);
                break;
            }
        }
        
        Ok(centroids)
    }
    
    fn assign_to_clusters(&self, vectors: &Tensor, centroids: &Tensor) -> Result<Vec<usize>> {
        // Compute distances to all centroids
        let distances = self.compute_distance_matrix(vectors, centroids)?;
        
        // Find closest centroid for each vector
        let cpu_distances = distances.to_device(&Device::Cpu)?;
        let distance_data = cpu_distances.to_vec2::<f32>()?;
        
        let assignments: Vec<usize> = distance_data
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();
        
        Ok(assignments)
    }
    
    fn update_centroids(&self, vectors: &Tensor, assignments: &[usize], k: usize) -> Result<Tensor> {
        let dimensions = vectors.shape()[1];
        let mut new_centroids = Vec::new();
        
        for cluster_id in 0..k {
            let cluster_vectors: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &assignment)| assignment == cluster_id)
                .map(|(idx, _)| idx)
                .collect();
            
            if cluster_vectors.is_empty() {
                // Keep old centroid if no assignments
                let old_centroid = vectors.get(cluster_id % vectors.shape()[0])?;
                new_centroids.push(old_centroid);
            } else {
                // Compute mean of assigned vectors
                let cluster_tensor = Tensor::stack(
                    &cluster_vectors.iter().map(|&i| vectors.get(i)).collect::<Result<Vec<_>, _>>()?,
                    0
                )?;
                let centroid = cluster_tensor.mean(0)?;
                new_centroids.push(centroid);
            }
        }
        
        Tensor::stack(&new_centroids, 0)
    }
    
    fn compute_distance_matrix(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Efficient squared distance computation
        let a_sq = a.sqr()?.sum_keepdim(1)?;
        let b_sq = b.sqr()?.sum_keepdim(1)?;
        let cross_term = a.matmul(&b.t()?)?.broadcast_mul(&Tensor::new(-2.0, &a.device())?)?;
        
        a_sq.broadcast_add(&b_sq.t()?)?.add(&cross_term)
    }
    
    fn compute_spiral_coordinates_gpu(
        &self,
        vectors: &Tensor,
        params: &CompressionParams,
    ) -> Result<Vec<(f32, f32)>> {
        // GPU-accelerated spiral coordinate computation
        let vector_count = vectors.shape()[0];
        
        // Compute vector norms and weighted sums
        let norms = vectors.sqr()?.sum(1)?.sqrt()?;
        let weighted_sums = self.compute_weighted_sums(vectors)?;
        
        // Transfer to CPU for coordinate generation
        let norms_cpu = norms.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let weighted_cpu = weighted_sums.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        
        let coords: Vec<(f32, f32)> = (0..vector_count)
            .map(|i| {
                let norm = norms_cpu[i];
                let weighted = weighted_cpu[i];
                
                let radius = norm * params.spiral_density;
                let angle = weighted / (norm + 1e-8) + i as f32 * 0.1;
                
                (radius * angle.cos(), radius * angle.sin())
            })
            .collect();
        
        Ok(coords)
    }
    
    fn compute_weighted_sums(&self, vectors: &Tensor) -> Result<Tensor> {
        let dimensions = vectors.shape()[1];
        let weights = Tensor::arange(1f32, dimensions as f32 + 1.0, &vectors.device())?;
        vectors.broadcast_mul(&weights.unsqueeze(0)?)?.sum(1)
    }
    
    fn generate_spiral_ordering(&self, coords: &[(f32, f32)]) -> Result<Vec<usize>> {
        let mut indexed_coords: Vec<(usize, f32)> = coords
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| (i, (x * x + y * y).sqrt()))
            .collect();
        
        // Sort by distance from origin (spiral ordering)
        indexed_coords.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        Ok(indexed_coords.into_iter().map(|(i, _)| i).collect())
    }
    
    fn generate_lsh_functions(
        &self,
        input_dim: usize,
        hash_dim: usize,
        num_hashes: usize,
    ) -> Result<Vec<Vec<f32>>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        (0..num_hashes)
            .map(|_| {
                (0..hash_dim * input_dim)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .map(Ok)
            .collect()
    }
    
    fn compute_vector_hashes(
        &self,
        vectors: &Tensor,
        hash_functions: &[Vec<f32>],
    ) -> Result<Vec<Vec<u64>>> {
        let vector_count = vectors.shape()[0];
        let cpu_vectors = vectors.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        
        let hashes: Vec<Vec<u64>> = cpu_vectors
            .iter()
            .map(|vector| {
                hash_functions
                    .iter()
                    .map(|hash_fn| self.compute_single_hash(vector, hash_fn))
                    .collect()
            })
            .collect();
        
        Ok(hashes)
    }
    
    fn compute_single_hash(&self, vector: &[f32], hash_function: &[f32]) -> u64 {
        let dimensions = vector.len();
        let hash_dim = hash_function.len() / dimensions;
        
        let mut hash_bits = 0u64;
        
        for h in 0..hash_dim.min(64) {
            let mut projection = 0.0;
            for d in 0..dimensions {
                projection += vector[d] * hash_function[h * dimensions + d];
            }
            
            if projection > 0.0 {
                hash_bits |= 1u64 << h;
            }
        }
        
        hash_bits
    }
    
    fn build_hash_tables(&self, vector_hashes: &[Vec<u64>]) -> Result<Vec<HashMap<u64, Vec<usize>>>> {
        let num_hashes = vector_hashes[0].len();
        let mut hash_tables = vec![HashMap::new(); num_hashes];
        
        for (vector_idx, hashes) in vector_hashes.iter().enumerate() {
            for (hash_idx, &hash_value) in hashes.iter().enumerate() {
                hash_tables[hash_idx]
                    .entry(hash_value)
                    .or_insert_with(Vec::new)
                    .push(vector_idx);
            }
        }
        
        Ok(hash_tables)
    }
    
    fn compute_cosine_similarity(&self, a: &Tensor, b: &Tensor) -> Result<f32> {
        let dot_product = a.mul(b)?.sum_all()?.to_scalar::<f32>()?;
        let norm_a = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let norm_b = b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        
        Ok(dot_product / (norm_a * norm_b + 1e-8))
    }
    
    fn tensor_to_vectors(&self, tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
        let cpu_tensor = tensor.to_device(&Device::Cpu)?;
        cpu_tensor.to_vec2::<f32>()
    }
}