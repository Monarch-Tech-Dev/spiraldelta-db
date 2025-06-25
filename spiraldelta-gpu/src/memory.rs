/*!
GPU Memory Management for SpiralDelta

Efficient memory allocation, pooling, and transfer strategies for optimal GPU performance.
*/

use cudarc::driver::{CudaDevice, CudaSlice};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use anyhow::{Result, anyhow};
use crate::{GpuConfig, GpuError};

pub struct GpuMemoryManager {
    device: Arc<CudaDevice>,
    config: GpuConfig,
    memory_pools: Arc<Mutex<HashMap<String, MemoryPool>>>,
    allocated_bytes: Arc<Mutex<usize>>,
}

struct MemoryPool {
    free_blocks: Vec<GpuMemoryBlock>,
    used_blocks: Vec<GpuMemoryBlock>,
    block_size: usize,
    max_blocks: usize,
}

#[derive(Clone)]
struct GpuMemoryBlock {
    ptr: CudaSlice<u8>,
    size: usize,
    in_use: bool,
}

impl GpuMemoryManager {
    pub fn new(device: Arc<CudaDevice>, config: &GpuConfig) -> Result<Self> {
        let max_memory_bytes = (config.memory_limit_gb * 1e9) as usize;
        let available_memory = device.available_memory()
            .map_err(|e| anyhow!("Failed to get available memory: {}", e))?;
        
        if max_memory_bytes > available_memory {
            return Err(anyhow!(
                "Requested memory ({:.2} GB) exceeds available memory ({:.2} GB)",
                config.memory_limit_gb,
                available_memory as f64 / 1e9
            ));
        }
        
        Ok(Self {
            device,
            config: config.clone(),
            memory_pools: Arc::new(Mutex::new(HashMap::new())),
            allocated_bytes: Arc::new(Mutex::new(0)),
        })
    }
    
    pub fn allocate_vectors(&self, vector_count: usize, dimensions: usize) -> Result<CudaSlice<f32>> {
        let total_elements = vector_count * dimensions;
        let bytes_needed = total_elements * std::mem::size_of::<f32>();
        
        self.check_memory_availability(bytes_needed)?;
        
        let slice = self.device.alloc_zeros::<f32>(total_elements)
            .map_err(|e| anyhow!("Failed to allocate GPU memory: {}", e))?;
        
        // Update allocated bytes counter
        {
            let mut allocated = self.allocated_bytes.lock().unwrap();
            *allocated += bytes_needed;
        }
        
        Ok(slice)
    }
    
    pub fn allocate_indices(&self, count: usize) -> Result<CudaSlice<i32>> {
        let bytes_needed = count * std::mem::size_of::<i32>();
        self.check_memory_availability(bytes_needed)?;
        
        let slice = self.device.alloc_zeros::<i32>(count)
            .map_err(|e| anyhow!("Failed to allocate GPU memory: {}", e))?;
        
        {
            let mut allocated = self.allocated_bytes.lock().unwrap();
            *allocated += bytes_needed;
        }
        
        Ok(slice)
    }
    
    pub fn transfer_to_gpu(&self, data: &[f32]) -> Result<CudaSlice<f32>> {
        let bytes_needed = data.len() * std::mem::size_of::<f32>();
        self.check_memory_availability(bytes_needed)?;
        
        let slice = self.device.htod_copy(data.to_vec())
            .map_err(|e| anyhow!("Failed to transfer data to GPU: {}", e))?;
        
        {
            let mut allocated = self.allocated_bytes.lock().unwrap();
            *allocated += bytes_needed;
        }
        
        Ok(slice)
    }
    
    pub fn transfer_from_gpu(&self, gpu_data: &CudaSlice<f32>) -> Result<Vec<f32>> {
        self.device.dtoh_sync_copy(gpu_data)
            .map_err(|e| anyhow!("Failed to transfer data from GPU: {}", e))
    }
    
    pub fn get_memory_pool(&self, pool_name: &str, block_size: usize, max_blocks: usize) -> Result<()> {
        let mut pools = self.memory_pools.lock().unwrap();
        
        if !pools.contains_key(pool_name) {
            let pool = MemoryPool {
                free_blocks: Vec::new(),
                used_blocks: Vec::new(),
                block_size,
                max_blocks,
            };
            pools.insert(pool_name.to_string(), pool);
        }
        
        Ok(())
    }
    
    pub fn allocate_from_pool(&self, pool_name: &str) -> Result<GpuMemoryBlock> {
        let mut pools = self.memory_pools.lock().unwrap();
        let pool = pools.get_mut(pool_name)
            .ok_or_else(|| anyhow!("Memory pool '{}' not found", pool_name))?;
        
        // Try to reuse a free block
        if let Some(mut block) = pool.free_blocks.pop() {
            block.in_use = true;
            pool.used_blocks.push(block.clone());
            return Ok(block);
        }
        
        // Allocate new block if under limit
        if pool.used_blocks.len() < pool.max_blocks {
            self.check_memory_availability(pool.block_size)?;
            
            let slice = self.device.alloc_zeros::<u8>(pool.block_size)
                .map_err(|e| anyhow!("Failed to allocate memory block: {}", e))?;
            
            let block = GpuMemoryBlock {
                ptr: slice,
                size: pool.block_size,
                in_use: true,
            };
            
            pool.used_blocks.push(block.clone());
            
            {
                let mut allocated = self.allocated_bytes.lock().unwrap();
                *allocated += pool.block_size;
            }
            
            return Ok(block);
        }
        
        Err(anyhow!("Memory pool '{}' exhausted", pool_name))
    }
    
    pub fn return_to_pool(&self, pool_name: &str, block: GpuMemoryBlock) -> Result<()> {
        let mut pools = self.memory_pools.lock().unwrap();
        let pool = pools.get_mut(pool_name)
            .ok_or_else(|| anyhow!("Memory pool '{}' not found", pool_name))?;
        
        // Find and remove from used blocks
        if let Some(pos) = pool.used_blocks.iter().position(|b| {
            std::ptr::eq(b.ptr.as_ptr(), block.ptr.as_ptr())
        }) {
            pool.used_blocks.remove(pos);
            
            // Add to free blocks for reuse
            let mut free_block = block;
            free_block.in_use = false;
            pool.free_blocks.push(free_block);
        }
        
        Ok(())
    }
    
    pub fn cleanup(&self) -> Result<()> {
        // Clear all memory pools
        {
            let mut pools = self.memory_pools.lock().unwrap();
            pools.clear();
        }
        
        // Reset allocated bytes counter
        {
            let mut allocated = self.allocated_bytes.lock().unwrap();
            *allocated = 0;
        }
        
        // Force GPU garbage collection
        self.device.synchronize()
            .map_err(|e| anyhow!("Failed to synchronize device: {}", e))?;
        
        Ok(())
    }
    
    pub fn get_memory_stats(&self) -> MemoryStats {
        let allocated = *self.allocated_bytes.lock().unwrap();
        let available = self.device.available_memory().unwrap_or(0);
        let total = self.device.total_memory().unwrap_or(0);
        
        MemoryStats {
            allocated_bytes: allocated,
            available_bytes: available,
            total_bytes: total,
            utilization_percent: (allocated as f64 / total as f64) * 100.0,
        }
    }
    
    fn check_memory_availability(&self, bytes_needed: usize) -> Result<()> {
        let available = self.device.available_memory()
            .map_err(|e| anyhow!("Failed to check available memory: {}", e))?;
        
        if bytes_needed > available {
            return Err(anyhow!(
                "Insufficient GPU memory: need {} bytes, available {} bytes",
                bytes_needed, available
            ));
        }
        
        // Check against our configured limit
        let current_allocated = *self.allocated_bytes.lock().unwrap();
        let max_allowed = (self.config.memory_limit_gb * 1e9) as usize;
        
        if current_allocated + bytes_needed > max_allowed {
            return Err(anyhow!(
                "Memory allocation would exceed configured limit: {} + {} > {}",
                current_allocated, bytes_needed, max_allowed
            ));
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub available_bytes: usize,
    pub total_bytes: usize,
    pub utilization_percent: f64,
}

impl MemoryStats {
    pub fn allocated_gb(&self) -> f64 {
        self.allocated_bytes as f64 / 1e9
    }
    
    pub fn available_gb(&self) -> f64 {
        self.available_bytes as f64 / 1e9
    }
    
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / 1e9
    }
}

pub struct GpuStreamManager {
    streams: Vec<cudarc::driver::CudaStream>,
    current_stream: usize,
}

impl GpuStreamManager {
    pub fn new(device: &CudaDevice, stream_count: usize) -> Result<Self> {
        let streams: Result<Vec<_>, _> = (0..stream_count)
            .map(|_| device.fork_default_stream())
            .collect();
        
        let streams = streams
            .map_err(|e| anyhow!("Failed to create CUDA streams: {}", e))?;
        
        Ok(Self {
            streams,
            current_stream: 0,
        })
    }
    
    pub fn get_next_stream(&mut self) -> &cudarc::driver::CudaStream {
        let stream = &self.streams[self.current_stream];
        self.current_stream = (self.current_stream + 1) % self.streams.len();
        stream
    }
    
    pub fn synchronize_all(&self) -> Result<()> {
        for stream in &self.streams {
            stream.synchronize()
                .map_err(|e| anyhow!("Failed to synchronize stream: {}", e))?;
        }
        Ok(())
    }
}