/*!
SpiralDelta GPU Acceleration Module

This module provides CUDA-accelerated implementations of core SpiralDeltaDB operations:
- Vector similarity search with massive parallelization
- Index construction and optimization
- Batch vector operations
- Memory-efficient data transfers

Performance targets:
- 100x speedup for similarity search on large datasets
- 10x faster index construction
- Sub-millisecond queries on millions of vectors
*/

use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor, DType};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use log::{info, warn, debug};

pub mod kernels;
pub mod memory;
pub mod search;
pub mod index;
pub mod spiral;

use kernels::*;
use memory::*;
use search::*;
use index::*;
use spiral::*;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("CUDA device not available: {0}")]
    CudaUnavailable(String),
    
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),
    
    #[error("Kernel execution failed: {0}")]
    KernelError(String),
    
    #[error("Data transfer failed: {0}")]
    TransferError(String),
    
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub device_id: i32,
    pub max_batch_size: usize,
    pub memory_limit_gb: f32,
    pub enable_mixed_precision: bool,
    pub enable_tensor_cores: bool,
    pub stream_count: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            max_batch_size: 10000,
            memory_limit_gb: 8.0,
            enable_mixed_precision: true,
            enable_tensor_cores: true,
            stream_count: 4,
        }
    }
}

/// Main GPU acceleration interface for SpiralDeltaDB
#[pyclass]
pub struct GpuAccelerator {
    device: Arc<CudaDevice>,
    config: GpuConfig,
    memory_manager: GpuMemoryManager,
    kernel_cache: HashMap<String, CompiledKernel>,
    stats: Arc<Mutex<GpuStats>>,
}

#[derive(Debug, Default)]
struct GpuStats {
    total_queries: u64,
    total_gpu_time_ms: f64,
    total_transfer_time_ms: f64,
    cache_hits: u64,
    cache_misses: u64,
}

#[pymethods]
impl GpuAccelerator {
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<GpuConfig>) -> PyResult<Self> {
        let config = config.unwrap_or_default();
        
        // Initialize CUDA device
        let device = Arc::new(
            CudaDevice::new(config.device_id)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to initialize CUDA device {}: {}", config.device_id, e)
                ))?
        );
        
        info!("Initialized GPU accelerator on device {}", config.device_id);
        info!("Device memory: {:.2} GB", device.total_memory()? as f64 / 1e9);
        
        let memory_manager = GpuMemoryManager::new(device.clone(), &config)?;
        
        Ok(Self {
            device,
            config,
            memory_manager,
            kernel_cache: HashMap::new(),
            stats: Arc::new(Mutex::new(GpuStats::default())),
        })
    }
    
    /// Check if GPU acceleration is available
    #[staticmethod]
    pub fn is_available() -> bool {
        cudarc::driver::CudaDevice::new(0).is_ok()
    }
    
    /// Get GPU device information
    pub fn get_device_info(&self) -> PyResult<HashMap<String, PyObject>> {
        let py = Python::acquire_gil();
        let py = py.python();
        
        let mut info = HashMap::new();
        info.insert("device_id".to_string(), self.config.device_id.to_object(py));
        info.insert("total_memory_gb".to_string(), 
                   (self.device.total_memory()? as f64 / 1e9).to_object(py));
        info.insert("free_memory_gb".to_string(), 
                   (self.device.available_memory()? as f64 / 1e9).to_object(py));
        
        let stats = self.stats.lock().unwrap();
        info.insert("total_queries".to_string(), stats.total_queries.to_object(py));
        info.insert("avg_gpu_time_ms".to_string(), 
                   (stats.total_gpu_time_ms / stats.total_queries as f64).to_object(py));
        info.insert("cache_hit_rate".to_string(), 
                   (stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64).to_object(py));
        
        Ok(info)
    }
    
    /// Perform GPU-accelerated vector similarity search
    #[pyo3(signature = (query_vectors, index_vectors, k = 10, metric = "cosine"))]
    pub fn similarity_search(
        &mut self,
        query_vectors: Vec<Vec<f32>>,
        index_vectors: Vec<Vec<f32>>,
        k: usize,
        metric: &str,
    ) -> PyResult<Vec<Vec<(usize, f32)>>> {
        let start_time = std::time::Instant::now();
        
        // Validate inputs
        if query_vectors.is_empty() || index_vectors.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Query and index vectors cannot be empty"
            ));
        }
        
        let dim = query_vectors[0].len();
        if !query_vectors.iter().all(|v| v.len() == dim) || 
           !index_vectors.iter().all(|v| v.len() == dim) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All vectors must have the same dimension"
            ));
        }
        
        // Convert to GPU format
        let query_tensor = self.vectors_to_tensor(&query_vectors)?;
        let index_tensor = self.vectors_to_tensor(&index_vectors)?;
        
        // Perform search on GPU
        let results = match metric {
            "cosine" => self.cosine_similarity_search(&query_tensor, &index_tensor, k)?,
            "euclidean" => self.euclidean_distance_search(&query_tensor, &index_tensor, k)?,
            "dot" => self.dot_product_search(&query_tensor, &index_tensor, k)?,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported metric: {}", metric)
            )),
        };
        
        // Update statistics
        let elapsed = start_time.elapsed().as_millis() as f64;
        let mut stats = self.stats.lock().unwrap();
        stats.total_queries += 1;
        stats.total_gpu_time_ms += elapsed;
        
        debug!("GPU similarity search completed in {:.2}ms", elapsed);
        
        Ok(results)
    }
    
    /// Build GPU-optimized index for fast searching
    pub fn build_index(
        &mut self,
        vectors: Vec<Vec<f32>>,
        index_type: &str,
    ) -> PyResult<PyObject> {
        let start_time = std::time::Instant::now();
        
        let py = Python::acquire_gil();
        let py = py.python();
        
        let tensor = self.vectors_to_tensor(&vectors)?;
        
        let index = match index_type {
            "hnsw" => self.build_hnsw_index(&tensor)?,
            "ivf" => self.build_ivf_index(&tensor)?,
            "spiral" => self.build_spiral_index(&tensor)?,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported index type: {}", index_type)
            )),
        };
        
        let elapsed = start_time.elapsed().as_millis() as f64;
        info!("GPU index construction completed in {:.2}ms", elapsed);
        
        // Return serialized index data
        Ok(bincode::serialize(&index)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            .to_object(py))
    }
    
    /// Perform batch vector operations with GPU acceleration
    pub fn batch_operations(
        &mut self,
        vectors: Vec<Vec<f32>>,
        operation: &str,
        params: Option<HashMap<String, f32>>,
    ) -> PyResult<Vec<Vec<f32>>> {
        let tensor = self.vectors_to_tensor(&vectors)?;
        
        let result_tensor = match operation {
            "normalize" => self.normalize_vectors(&tensor)?,
            "pca_reduce" => {
                let target_dim = params
                    .as_ref()
                    .and_then(|p| p.get("target_dim"))
                    .map(|&d| d as usize)
                    .unwrap_or(128);
                self.pca_reduce(&tensor, target_dim)?
            },
            "quantize" => {
                let bits = params
                    .as_ref()
                    .and_then(|p| p.get("bits"))
                    .map(|&b| b as u8)
                    .unwrap_or(8);
                self.quantize_vectors(&tensor, bits)?
            },
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unsupported operation: {}", operation)
            )),
        };
        
        self.tensor_to_vectors(&result_tensor)
    }
    
    /// Optimize memory usage and clear caches
    pub fn optimize_memory(&mut self) -> PyResult<()> {
        self.memory_manager.cleanup()?;
        self.kernel_cache.clear();
        
        // Force garbage collection on GPU
        self.device.synchronize()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        info!("GPU memory optimized");
        Ok(())
    }
}

impl GpuAccelerator {
    fn vectors_to_tensor(&self, vectors: &[Vec<f32>]) -> Result<Tensor> {
        let rows = vectors.len();
        let cols = vectors[0].len();
        
        let flat_data: Vec<f32> = vectors.iter().flatten().copied().collect();
        
        Tensor::from_vec(flat_data, (rows, cols), &Device::Cuda(self.device.clone()))
            .map_err(|e| anyhow!("Failed to create tensor: {}", e))
    }
    
    fn tensor_to_vectors(&self, tensor: &Tensor) -> PyResult<Vec<Vec<f32>>> {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Tensor must be 2-dimensional"
            ));
        }
        
        let data: Vec<f32> = tensor
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        
        let rows = shape[0];
        let cols = shape[1];
        
        Ok(data.chunks(cols).map(|chunk| chunk.to_vec()).collect())
    }
    
    fn cosine_similarity_search(
        &self,
        query: &Tensor,
        index: &Tensor,
        k: usize,
    ) -> Result<Vec<Vec<(usize, f32)>>> {
        // Normalize vectors
        let query_norm = query.broadcast_div(&query.sum_keepdim(1)?)?;
        let index_norm = index.broadcast_div(&index.sum_keepdim(1)?)?;
        
        // Compute similarity matrix
        let similarities = query_norm.matmul(&index_norm.t()?)?;
        
        // Get top-k results
        self.extract_topk(&similarities, k)
    }
    
    fn euclidean_distance_search(
        &self,
        query: &Tensor,
        index: &Tensor,
        k: usize,
    ) -> Result<Vec<Vec<(usize, f32)>>> {
        // Compute squared distances efficiently
        let query_sq = query.sqr()?.sum_keepdim(1)?;
        let index_sq = index.sqr()?.sum_keepdim(1)?;
        let cross_term = query.matmul(&index.t()?)?.broadcast_mul(&Tensor::new(-2.0, &query.device())?)?;
        
        let distances = query_sq.broadcast_add(&index_sq.t()?)?.add(&cross_term)?;
        
        // Convert to negative distances for top-k (we want smallest distances)
        let neg_distances = distances.neg()?;
        self.extract_topk(&neg_distances, k)
    }
    
    fn dot_product_search(
        &self,
        query: &Tensor,
        index: &Tensor,
        k: usize,
    ) -> Result<Vec<Vec<(usize, f32)>>> {
        let dot_products = query.matmul(&index.t()?)?;
        self.extract_topk(&dot_products, k)
    }
    
    fn extract_topk(&self, similarities: &Tensor, k: usize) -> Result<Vec<Vec<(usize, f32)>>> {
        // Move to CPU for top-k extraction (could be optimized with GPU kernels)
        let cpu_similarities = similarities.to_device(&Device::Cpu)?;
        let sim_data = cpu_similarities.to_vec2::<f32>()?;
        
        let results: Vec<Vec<(usize, f32)>> = sim_data
            .par_iter()
            .map(|row| {
                let mut indexed: Vec<(usize, f32)> = row
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| (i, val))
                    .collect();
                
                // Sort by similarity (descending)
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed.truncate(k);
                indexed
            })
            .collect();
        
        Ok(results)
    }
    
    fn normalize_vectors(&self, vectors: &Tensor) -> Result<Tensor> {
        let norms = vectors.sqr()?.sum_keepdim(1)?.sqrt()?;
        vectors.broadcast_div(&norms).map_err(|e| anyhow!("Normalization failed: {}", e))
    }
    
    fn pca_reduce(&self, vectors: &Tensor, target_dim: usize) -> Result<Tensor> {
        // Simplified PCA using SVD
        // In production, would use more sophisticated GPU-optimized PCA
        let (u, _s, _v) = vectors.svd()?;
        let reduced = u.narrow(1, 0, target_dim)?;
        Ok(reduced)
    }
    
    fn quantize_vectors(&self, vectors: &Tensor, _bits: u8) -> Result<Tensor> {
        // Simple quantization - could be made more sophisticated
        let scale = 255.0f32;
        let quantized = vectors.broadcast_mul(&Tensor::new(scale, &vectors.device())?)?
            .round()?
            .broadcast_div(&Tensor::new(scale, &vectors.device())?)?;
        Ok(quantized)
    }
    
    fn build_hnsw_index(&self, _vectors: &Tensor) -> Result<GpuIndex> {
        // Placeholder for HNSW implementation
        Ok(GpuIndex::HNSW { layers: vec![] })
    }
    
    fn build_ivf_index(&self, _vectors: &Tensor) -> Result<GpuIndex> {
        // Placeholder for IVF implementation
        Ok(GpuIndex::IVF { centroids: vec![], assignments: vec![] })
    }
    
    fn build_spiral_index(&self, vectors: &Tensor) -> Result<GpuIndex> {
        // GPU-accelerated spiral ordering
        let spiral_coords = self.compute_spiral_coordinates(vectors)?;
        Ok(GpuIndex::Spiral { coordinates: spiral_coords, order: vec![] })
    }
    
    fn compute_spiral_coordinates(&self, vectors: &Tensor) -> Result<Vec<(f32, f32)>> {
        // GPU-accelerated spiral coordinate computation
        // This would use custom CUDA kernels for optimal performance
        let cpu_vectors = vectors.to_device(&Device::Cpu)?;
        let data = cpu_vectors.to_vec2::<f32>()?;
        
        // Simplified spiral computation - would be optimized with CUDA kernels
        let coords: Vec<(f32, f32)> = data
            .par_iter()
            .enumerate()
            .map(|(i, row)| {
                let sum = row.iter().sum::<f32>();
                let avg = sum / row.len() as f32;
                let angle = (i as f32 * 0.1) % (2.0 * std::f32::consts::PI);
                let radius = avg.abs();
                (radius * angle.cos(), radius * angle.sin())
            })
            .collect();
        
        Ok(coords)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum GpuIndex {
    HNSW { layers: Vec<Vec<usize>> },
    IVF { centroids: Vec<Vec<f32>>, assignments: Vec<usize> },
    Spiral { coordinates: Vec<(f32, f32)>, order: Vec<usize> },
}

/// Python module definition
#[pymodule]
fn spiraldelta_gpu(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::init();
    
    m.add_class::<GpuAccelerator>()?;
    m.add_function(wrap_pyfunction!(check_cuda_availability, m)?)?;
    m.add_function(wrap_pyfunction!(get_gpu_memory_info, m)?)?;
    
    Ok(())
}

#[pyfunction]
fn check_cuda_availability() -> bool {
    GpuAccelerator::is_available()
}

#[pyfunction]
fn get_gpu_memory_info() -> PyResult<HashMap<String, f64>> {
    if let Ok(device) = cudarc::driver::CudaDevice::new(0) {
        let py = Python::acquire_gil();
        let py = py.python();
        
        let mut info = HashMap::new();
        info.insert("total_gb".to_string(), device.total_memory().unwrap_or(0) as f64 / 1e9);
        info.insert("available_gb".to_string(), device.available_memory().unwrap_or(0) as f64 / 1e9);
        
        Ok(info)
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "CUDA device not available"
        ))
    }
}