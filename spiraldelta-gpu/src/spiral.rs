/*!
GPU-Accelerated Spiral Ordering and Compression

Advanced spiral coordinate computation and delta compression optimized for GPU execution.
*/

use candle_core::{Device, Tensor};
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use crate::memory::GpuMemoryManager;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiralParameters {
    pub spiral_constant: f32,
    pub compression_threshold: f32,
    pub delta_precision: f32,
    pub adaptive_scaling: bool,
    pub multi_scale_levels: usize,
}

impl Default for SpiralParameters {
    fn default() -> Self {
        Self {
            spiral_constant: 0.1,
            compression_threshold: 0.01,
            delta_precision: 1e-6,
            adaptive_scaling: true,
            multi_scale_levels: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiralIndex {
    pub coordinates: Vec<(f32, f32)>,
    pub ordering: Vec<usize>,
    pub compressed_deltas: Vec<CompressedDelta>,
    pub scale_factors: Vec<f32>,
    pub parameters: SpiralParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedDelta {
    pub base_index: usize,
    pub delta_vector: Vec<f32>,
    pub compression_ratio: f32,
    pub reconstruction_error: f32,
}

pub struct GpuSpiralEngine {
    device: Arc<CudaDevice>,
    memory_manager: Arc<GpuMemoryManager>,
}

impl GpuSpiralEngine {
    pub fn new(device: Arc<CudaDevice>, memory_manager: Arc<GpuMemoryManager>) -> Self {
        Self {
            device,
            memory_manager,
        }
    }
    
    /// Compute spiral coordinates with GPU acceleration
    pub fn compute_spiral_coordinates(
        &self,
        vectors: &Tensor,
        params: &SpiralParameters,
    ) -> Result<Vec<(f32, f32)>> {
        let vector_count = vectors.shape()[0];
        let dimensions = vectors.shape()[1];
        
        log::info!("Computing spiral coordinates for {} vectors of {} dimensions", vector_count, dimensions);
        
        if params.adaptive_scaling {
            self.adaptive_spiral_coordinates(vectors, params)
        } else {
            self.fixed_spiral_coordinates(vectors, params)
        }
    }
    
    fn adaptive_spiral_coordinates(
        &self,
        vectors: &Tensor,
        params: &SpiralParameters,
    ) -> Result<Vec<(f32, f32)>> {
        // Multi-scale spiral coordinate computation
        let mut all_coordinates = Vec::new();
        
        for scale_level in 0..params.multi_scale_levels {
            let scale_factor = 2.0f32.powi(scale_level as i32);
            let level_coords = self.compute_scale_level_coordinates(vectors, params, scale_factor)?;
            all_coordinates.push(level_coords);
        }
        
        // Combine coordinates from different scales
        self.combine_multi_scale_coordinates(all_coordinates)
    }
    
    fn compute_scale_level_coordinates(
        &self,
        vectors: &Tensor,
        params: &SpiralParameters,
        scale_factor: f32,
    ) -> Result<Vec<(f32, f32)>> {
        let vector_count = vectors.shape()[0];
        
        // Compute primary spiral attributes on GPU
        let (radii, angles) = self.compute_spiral_attributes_gpu(vectors, params, scale_factor)?;
        
        // Convert to Cartesian coordinates
        let coordinates: Vec<(f32, f32)> = (0..vector_count)
            .map(|i| {
                let r = radii[i];
                let theta = angles[i];
                (r * theta.cos(), r * theta.sin())
            })
            .collect();
        
        Ok(coordinates)
    }
    
    fn compute_spiral_attributes_gpu(
        &self,
        vectors: &Tensor,
        params: &SpiralParameters,
        scale_factor: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // Compute vector magnitudes
        let magnitudes = vectors.sqr()?.sum(1)?.sqrt()?;
        
        // Compute directional complexity (weighted sum of components)
        let complexity = self.compute_directional_complexity(vectors)?;
        
        // Compute spiral angles based on vector properties
        let angles = self.compute_spiral_angles(vectors, &complexity, params)?;
        
        // Transfer results to CPU
        let mag_cpu = magnitudes.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let ang_cpu = angles.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        
        // Apply scale factor to radii
        let radii: Vec<f32> = mag_cpu.iter().map(|&m| m * scale_factor).collect();
        
        Ok((radii, ang_cpu))
    }
    
    fn compute_directional_complexity(&self, vectors: &Tensor) -> Result<Tensor> {
        let dimensions = vectors.shape()[1];
        
        // Create dimension weights (higher dimensions get more weight)
        let dim_weights = Tensor::arange(1f32, dimensions as f32 + 1.0, &vectors.device())?
            .powf(0.5)?; // Square root weighting
        
        // Compute weighted sum of absolute values
        let abs_vectors = vectors.abs()?;
        let complexity = abs_vectors.broadcast_mul(&dim_weights.unsqueeze(0)?)?.sum(1)?;
        
        Ok(complexity)
    }
    
    fn compute_spiral_angles(
        &self,
        vectors: &Tensor,
        complexity: &Tensor,
        params: &SpiralParameters,
    ) -> Result<Tensor> {
        let vector_count = vectors.shape()[0];
        
        // Normalize complexity for angle computation
        let max_complexity = complexity.max(0)?.to_scalar::<f32>()?;
        let normalized_complexity = complexity.broadcast_div(&Tensor::new(max_complexity + 1e-8, &vectors.device())?)?;
        
        // Compute base angles from vector properties
        let base_angles = self.compute_base_angles(vectors)?;
        
        // Add spiral progression
        let indices = Tensor::arange(0f32, vector_count as f32, &vectors.device())?;
        let spiral_progression = indices.broadcast_mul(&Tensor::new(params.spiral_constant, &vectors.device())?)?;
        
        // Combine base angles with spiral progression and complexity
        let combined_angles = base_angles
            .add(&spiral_progression)?
            .add(&normalized_complexity.broadcast_mul(&Tensor::new(std::f32::consts::PI, &vectors.device())?)?)?;
        
        Ok(combined_angles)
    }
    
    fn compute_base_angles(&self, vectors: &Tensor) -> Result<Tensor> {
        let dimensions = vectors.shape()[1];
        
        if dimensions < 2 {
            return Ok(Tensor::zeros_like(&vectors.narrow(1, 0, 1)?.squeeze(1)?)?);
        }
        
        // Use first two principal components for base angle
        let x_component = vectors.narrow(1, 0, 1)?.squeeze(1)?;
        let y_component = vectors.narrow(1, 1, 1)?.squeeze(1)?;
        
        // Compute atan2 equivalent
        self.atan2_gpu(&y_component, &x_component)
    }
    
    fn atan2_gpu(&self, y: &Tensor, x: &Tensor) -> Result<Tensor> {
        // GPU-friendly atan2 implementation
        let pi = Tensor::new(std::f32::consts::PI, &y.device())?;
        let half_pi = Tensor::new(std::f32::consts::PI / 2.0, &y.device())?;
        
        let abs_x = x.abs()?;
        let abs_y = y.abs()?;
        
        // Compute atan(y/x) approximation
        let ratio = y.div(&(x.abs().add(&Tensor::new(1e-8, &y.device())?)?)?)?;
        let atan_approx = ratio.div(&(Tensor::new(1.0, &y.device())?.add(&ratio.abs())?))?;
        
        // Adjust for quadrant
        let angle = Tensor::where_cond(
            &x.ge(&Tensor::new(0.0, &x.device())?)?,
            &atan_approx,
            &Tensor::where_cond(
                &y.ge(&Tensor::new(0.0, &y.device())?)?,
                &pi.sub(&atan_approx)?,
                &pi.neg()?.add(&atan_approx)?
            )?
        )?;
        
        Ok(angle)
    }
    
    fn combine_multi_scale_coordinates(
        &self,
        scale_coordinates: Vec<Vec<(f32, f32)>>,
    ) -> Result<Vec<(f32, f32)>> {
        if scale_coordinates.is_empty() {
            return Ok(Vec::new());
        }
        
        let vector_count = scale_coordinates[0].len();
        let mut combined = Vec::with_capacity(vector_count);
        
        for i in 0..vector_count {
            let mut x_sum = 0.0;
            let mut y_sum = 0.0;
            let mut weight_sum = 0.0;
            
            for (scale_idx, coords) in scale_coordinates.iter().enumerate() {
                let weight = 1.0 / (scale_idx as f32 + 1.0); // Decreasing weights for higher scales
                x_sum += coords[i].0 * weight;
                y_sum += coords[i].1 * weight;
                weight_sum += weight;
            }
            
            combined.push((x_sum / weight_sum, y_sum / weight_sum));
        }
        
        Ok(combined)
    }
    
    fn fixed_spiral_coordinates(
        &self,
        vectors: &Tensor,
        params: &SpiralParameters,
    ) -> Result<Vec<(f32, f32)>> {
        let (radii, angles) = self.compute_spiral_attributes_gpu(vectors, params, 1.0)?;
        
        let coordinates: Vec<(f32, f32)> = radii
            .iter()
            .zip(angles.iter())
            .map(|(&r, &theta)| (r * theta.cos(), r * theta.sin()))
            .collect();
        
        Ok(coordinates)
    }
    
    /// Generate spiral ordering from coordinates
    pub fn generate_spiral_ordering(&self, coordinates: &[(f32, f32)]) -> Result<Vec<usize>> {
        // Sort by distance from origin (creating spiral ordering)
        let mut indexed_coords: Vec<(usize, f32)> = coordinates
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| (i, (x * x + y * y).sqrt()))
            .collect();
        
        indexed_coords.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(indexed_coords.into_iter().map(|(i, _)| i).collect())
    }
    
    /// Compute delta compression with GPU acceleration
    pub fn compute_delta_compression(
        &self,
        vectors: &Tensor,
        ordering: &[usize],
        params: &SpiralParameters,
    ) -> Result<Vec<CompressedDelta>> {
        let mut compressed_deltas = Vec::new();
        
        // Process vectors in spiral order for optimal compression
        for window in ordering.windows(2) {
            let base_idx = window[0];
            let current_idx = window[1];
            
            let compressed = self.compute_delta_pair(vectors, base_idx, current_idx, params)?;
            compressed_deltas.push(compressed);
        }
        
        Ok(compressed_deltas)
    }
    
    fn compute_delta_pair(
        &self,
        vectors: &Tensor,
        base_idx: usize,
        current_idx: usize,
        params: &SpiralParameters,
    ) -> Result<CompressedDelta> {
        let base_vector = vectors.get(base_idx)?;
        let current_vector = vectors.get(current_idx)?;
        
        // Compute delta vector
        let delta = current_vector.sub(&base_vector)?;
        
        // Apply compression threshold
        let compressed_delta = self.apply_compression_threshold(&delta, params)?;
        
        // Compute reconstruction error
        let reconstructed = base_vector.add(&compressed_delta)?;
        let error = current_vector.sub(&reconstructed)?.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        
        // Compute compression ratio
        let original_norm = delta.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let compressed_norm = compressed_delta.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let compression_ratio = if original_norm > 0.0 {
            1.0 - (compressed_norm / original_norm)
        } else {
            0.0
        };
        
        let delta_vec = compressed_delta.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        
        Ok(CompressedDelta {
            base_index: base_idx,
            delta_vector: delta_vec,
            compression_ratio,
            reconstruction_error: error,
        })
    }
    
    fn apply_compression_threshold(&self, delta: &Tensor, params: &SpiralParameters) -> Result<Tensor> {
        // Apply threshold-based compression
        let threshold = Tensor::new(params.compression_threshold, &delta.device())?;
        let abs_delta = delta.abs()?;
        
        // Zero out components below threshold
        let mask = abs_delta.gt(&threshold)?;
        let compressed = delta.broadcast_mul(&mask.to_dtype(delta.dtype())?)?;
        
        Ok(compressed)
    }
    
    /// Reconstruct vector from compressed representation
    pub fn reconstruct_vector(
        &self,
        base_vector: &[f32],
        compressed_delta: &CompressedDelta,
    ) -> Result<Vec<f32>> {
        if base_vector.len() != compressed_delta.delta_vector.len() {
            return Err(anyhow!("Base vector and delta dimensions don't match"));
        }
        
        let reconstructed: Vec<f32> = base_vector
            .iter()
            .zip(compressed_delta.delta_vector.iter())
            .map(|(&base, &delta)| base + delta)
            .collect();
        
        Ok(reconstructed)
    }
    
    /// Build complete spiral index
    pub fn build_spiral_index(
        &self,
        vectors: &Tensor,
        params: SpiralParameters,
    ) -> Result<SpiralIndex> {
        log::info!("Building spiral index with parameters: {:?}", params);
        
        // Compute spiral coordinates
        let coordinates = self.compute_spiral_coordinates(vectors, &params)?;
        
        // Generate spiral ordering
        let ordering = self.generate_spiral_ordering(&coordinates)?;
        
        // Compute delta compression
        let compressed_deltas = self.compute_delta_compression(vectors, &ordering, &params)?;
        
        // Compute scale factors for normalization
        let scale_factors = self.compute_scale_factors(vectors)?;
        
        Ok(SpiralIndex {
            coordinates,
            ordering,
            compressed_deltas,
            scale_factors,
            parameters: params,
        })
    }
    
    fn compute_scale_factors(&self, vectors: &Tensor) -> Result<Vec<f32>> {
        // Compute per-dimension scale factors for normalization
        let dimensions = vectors.shape()[1];
        let mut scale_factors = Vec::with_capacity(dimensions);
        
        for dim in 0..dimensions {
            let dim_values = vectors.narrow(1, dim, 1)?.squeeze(1)?;
            let std_dev = self.compute_std_dev(&dim_values)?;
            scale_factors.push(std_dev);
        }
        
        Ok(scale_factors)
    }
    
    fn compute_std_dev(&self, values: &Tensor) -> Result<f32> {
        let mean = values.mean(0)?.to_scalar::<f32>()?;
        let mean_tensor = Tensor::new(mean, &values.device())?;
        let variance = values.sub(&mean_tensor)?.sqr()?.mean(0)?.to_scalar::<f32>()?;
        Ok(variance.sqrt())
    }
    
    /// Search within spiral index
    pub fn spiral_search(
        &self,
        index: &SpiralIndex,
        query_coords: (f32, f32),
        k: usize,
        search_radius: f32,
    ) -> Result<Vec<usize>> {
        // Find candidates within search radius
        let candidates: Vec<usize> = index
            .coordinates
            .iter()
            .enumerate()
            .filter(|(_, &(x, y))| {
                let dist = ((x - query_coords.0).powi(2) + (y - query_coords.1).powi(2)).sqrt();
                dist <= search_radius
            })
            .map(|(i, _)| i)
            .collect();
        
        // Sort candidates by distance
        let mut candidate_distances: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|idx| {
                let (x, y) = index.coordinates[idx];
                let dist = ((x - query_coords.0).powi(2) + (y - query_coords.1).powi(2)).sqrt();
                (idx, dist)
            })
            .collect();
        
        candidate_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidate_distances.truncate(k);
        
        Ok(candidate_distances.into_iter().map(|(idx, _)| idx).collect())
    }
}