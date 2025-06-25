/*!
Distributed Worker Node

Handles query execution, data storage, and shard management for distributed operations.
*/

use crate::*;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, error, debug, instrument};
use serde::{Serialize, Deserialize};
use tokio::net::TcpListener;
use tonic::{transport::Server, Request, Response, Status};

/// Worker node that executes queries and manages data shards
pub struct WorkerNode {
    /// Node information
    node_info: NodeInfo,
    
    /// Node configuration
    config: WorkerConfig,
    
    /// Local shard storage
    local_shards: Arc<RwLock<HashMap<ShardId, LocalShard>>>,
    
    /// Query processor
    query_processor: Arc<QueryProcessor>,
    
    /// Storage engine
    storage_engine: Arc<StorageEngine>,
    
    /// Monitoring system
    monitoring: Arc<MonitoringSystem>,
    
    /// gRPC server for handling requests
    grpc_server: Option<tonic::transport::server::Router>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerConfig {
    /// Maximum memory per shard (GB)
    pub max_shard_memory_gb: f64,
    
    /// Maximum concurrent queries
    pub max_concurrent_queries: usize,
    
    /// Query timeout in milliseconds
    pub query_timeout_ms: u64,
    
    /// Enable compression for storage
    pub enable_compression: bool,
    
    /// Replication settings
    pub replication: ReplicationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// Enable automatic replication
    pub enabled: bool,
    
    /// Replication factor
    pub factor: usize,
    
    /// Sync mode (async/sync)
    pub sync_mode: ReplicationSyncMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicationSyncMode {
    Asynchronous,
    Synchronous,
    QuorumSync,
}

#[derive(Debug)]
struct LocalShard {
    shard_info: ShardInfo,
    vector_storage: VectorStorage,
    index: LocalIndex,
    statistics: ShardStatistics,
}

#[derive(Debug)]
struct VectorStorage {
    vectors: Vec<Vec<f32>>,
    metadata: Vec<VectorMetadata>,
    size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorMetadata {
    vector_id: u64,
    timestamp: chrono::DateTime<chrono::Utc>,
    tags: HashMap<String, String>,
}

#[derive(Debug)]
struct LocalIndex {
    index_type: IndexType,
    index_data: Vec<u8>, // Serialized index structure
    build_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum IndexType {
    Hnsw,
    IvfFlat,
    SpiralOptimized,
}

#[derive(Debug, Default, Clone)]
struct ShardStatistics {
    query_count: u64,
    last_query: Option<chrono::DateTime<chrono::Utc>>,
    avg_query_time_ms: f64,
    memory_usage_mb: f64,
    index_size_mb: f64,
}

/// Query processor for handling individual shard queries
pub struct QueryProcessor {
    worker_config: WorkerConfig,
    monitoring: Arc<MonitoringSystem>,
}

/// Storage engine for persistent data management
pub struct StorageEngine {
    storage_path: std::path::PathBuf,
    compression_enabled: bool,
    db_handle: Option<rocksdb::DB>,
}

impl WorkerNode {
    /// Create a new worker node
    pub async fn new(
        node_info: NodeInfo,
        config: WorkerConfig,
        monitoring: Arc<MonitoringSystem>,
    ) -> Result<Self> {
        let query_processor = Arc::new(QueryProcessor::new(config.clone(), monitoring.clone()));
        let storage_engine = Arc::new(StorageEngine::new(&config).await?);
        
        Ok(Self {
            node_info,
            config,
            local_shards: Arc::new(RwLock::new(HashMap::new())),
            query_processor,
            storage_engine,
            monitoring,
            grpc_server: None,
        })
    }
    
    /// Start the worker node server
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting worker node: {}", self.node_info.node_id);
        
        // Initialize storage
        self.storage_engine.initialize().await?;
        
        // Load existing shards from storage
        self.load_shards_from_storage().await?;
        
        // Start gRPC server
        self.start_grpc_server().await?;
        
        // Start background tasks
        self.start_background_tasks().await?;
        
        info!("Worker node {} started successfully", self.node_info.node_id);
        Ok(())
    }
    
    /// Execute a query on local shards
    #[instrument(skip(self, request))]
    pub async fn execute_shard_query(&self, request: ShardQueryRequest) -> Result<ShardQueryResponse> {
        let start_time = std::time::Instant::now();
        
        // Get the requested shard
        let local_shards = self.local_shards.read().await;
        let shard = local_shards.get(&request.shard_id)
            .ok_or_else(|| anyhow!("Shard {} not found on this worker", request.shard_id))?;
        
        // Execute query based on type
        let results = match request.query_type {
            QueryType::SimilaritySearch => {
                self.execute_similarity_search(shard, &request).await?
            }
            QueryType::VectorInsertion => {
                drop(local_shards); // Release read lock
                self.execute_vector_insertion(&request).await?
            }
            QueryType::IndexConstruction => {
                drop(local_shards); // Release read lock
                self.execute_index_construction(&request).await?
            }
            _ => vec![],
        };
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        // Update statistics
        self.update_shard_statistics(&request.shard_id, execution_time).await?;
        
        Ok(ShardQueryResponse {
            query_id: request.query_id,
            shard_id: request.shard_id,
            results,
            status: "success".to_string(),
            execution_time_ms: execution_time,
            error: None,
        })
    }
    
    async fn execute_similarity_search(
        &self,
        shard: &LocalShard,
        request: &ShardQueryRequest,
    ) -> Result<Vec<(usize, f32)>> {
        let k = request.parameters.k.unwrap_or(10);
        
        // Use query processor for optimized search
        self.query_processor.similarity_search(
            &request.query_vector,
            &shard.vector_storage,
            &shard.index,
            k,
        ).await
    }
    
    async fn execute_vector_insertion(&self, request: &ShardQueryRequest) -> Result<Vec<(usize, f32)>> {
        let mut local_shards = self.local_shards.write().await;
        let shard = local_shards.get_mut(&request.shard_id)
            .ok_or_else(|| anyhow!("Shard {} not found", request.shard_id))?;
        
        // Generate vector ID
        let vector_id = shard.vector_storage.vectors.len() as u64;
        
        // Add vector to storage
        shard.vector_storage.vectors.push(request.query_vector.clone());
        shard.vector_storage.metadata.push(VectorMetadata {
            vector_id,
            timestamp: chrono::Utc::now(),
            tags: request.parameters.extra_params.clone(),
        });
        
        // Update size tracking
        let vector_size = request.query_vector.len() * std::mem::size_of::<f32>();
        shard.vector_storage.size_bytes += vector_size as u64;
        shard.shard_info.vector_count += 1;
        shard.shard_info.size_bytes += vector_size as u64;
        
        // Mark index for rebuilding if needed
        if shard.vector_storage.vectors.len() % 1000 == 0 {
            // Trigger index rebuild every 1000 vectors
            self.schedule_index_rebuild(request.shard_id).await?;
        }
        
        // Persist to storage
        self.storage_engine.persist_vector(request.shard_id, vector_id, &request.query_vector).await?;
        
        Ok(vec![(vector_id as usize, 1.0)])
    }
    
    async fn execute_index_construction(&self, request: &ShardQueryRequest) -> Result<Vec<(usize, f32)>> {
        let mut local_shards = self.local_shards.write().await;
        let shard = local_shards.get_mut(&request.shard_id)
            .ok_or_else(|| anyhow!("Shard {} not found", request.shard_id))?;
        
        // Build optimized index for the shard
        let new_index = self.query_processor.build_index(&shard.vector_storage).await?;
        shard.index = new_index;
        
        // Update statistics
        shard.statistics.index_size_mb = 0.1 * shard.vector_storage.vectors.len() as f64; // Estimate
        
        Ok(vec![(1, 1.0)]) // Success indicator
    }
    
    async fn update_shard_statistics(&self, shard_id: &ShardId, execution_time_ms: u64) -> Result<()> {
        let mut local_shards = self.local_shards.write().await;
        if let Some(shard) = local_shards.get_mut(shard_id) {
            shard.statistics.query_count += 1;
            shard.statistics.last_query = Some(chrono::Utc::now());
            
            // Update running average
            let current_avg = shard.statistics.avg_query_time_ms;
            let count = shard.statistics.query_count as f64;
            shard.statistics.avg_query_time_ms = 
                (current_avg * (count - 1.0) + execution_time_ms as f64) / count;
        }
        Ok(())
    }
    
    async fn load_shards_from_storage(&self) -> Result<()> {
        // Load shard metadata from persistent storage
        let shard_ids = self.storage_engine.list_shards().await?;
        let mut local_shards = self.local_shards.write().await;
        
        for shard_id in shard_ids {
            let shard_data = self.storage_engine.load_shard(shard_id).await?;
            local_shards.insert(shard_id, shard_data);
        }
        
        info!("Loaded {} shards from storage", local_shards.len());
        Ok(())
    }
    
    async fn start_grpc_server(&mut self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.node_info.address.port()).parse()?;
        
        // Create gRPC service implementation
        let worker_service = WorkerService::new(
            self.local_shards.clone(),
            self.query_processor.clone(),
            self.monitoring.clone(),
        );
        
        // Start server in background
        let server = Server::builder()
            .add_service(worker_service.into_service())
            .serve(addr);
            
        tokio::spawn(async move {
            if let Err(e) = server.await {
                error!("gRPC server error: {}", e);
            }
        });
        
        info!("Worker gRPC server started on {}", addr);
        Ok(())
    }
    
    async fn start_background_tasks(&self) -> Result<()> {
        // Start health monitoring
        let monitoring = self.monitoring.clone();
        let node_id = self.node_info.node_id.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                if let Err(e) = monitoring.report_node_health(&node_id).await {
                    warn!("Failed to report node health: {}", e);
                }
            }
        });
        
        // Start periodic index optimization
        let local_shards = self.local_shards.clone();
        let query_processor = self.query_processor.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600)); // Hourly
            loop {
                interval.tick().await;
                
                let shards = local_shards.read().await;
                for (shard_id, shard) in shards.iter() {
                    // Check if index needs optimization
                    if shard.statistics.query_count > 10000 {
                        if let Err(e) = query_processor.optimize_index(*shard_id, &shard.vector_storage).await {
                            warn!("Failed to optimize index for shard {}: {}", shard_id, e);
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    async fn schedule_index_rebuild(&self, shard_id: ShardId) -> Result<()> {
        // Schedule index rebuild in background
        let local_shards = self.local_shards.clone();
        let query_processor = self.query_processor.clone();
        
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await; // Brief delay
            
            let shards = local_shards.read().await;
            if let Some(shard) = shards.get(&shard_id) {
                if let Err(e) = query_processor.rebuild_index(shard_id, &shard.vector_storage).await {
                    warn!("Failed to rebuild index for shard {}: {}", shard_id, e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Get worker node statistics
    pub async fn get_statistics(&self) -> Result<WorkerStatistics> {
        let local_shards = self.local_shards.read().await;
        
        let total_vectors: u64 = local_shards.values()
            .map(|shard| shard.shard_info.vector_count)
            .sum();
        
        let total_size_bytes: u64 = local_shards.values()
            .map(|shard| shard.shard_info.size_bytes)
            .sum();
        
        let total_queries: u64 = local_shards.values()
            .map(|shard| shard.statistics.query_count)
            .sum();
        
        let avg_query_time: f64 = if local_shards.len() > 0 {
            local_shards.values()
                .map(|shard| shard.statistics.avg_query_time_ms)
                .sum::<f64>() / local_shards.len() as f64
        } else {
            0.0
        };
        
        Ok(WorkerStatistics {
            node_id: self.node_info.node_id.clone(),
            shard_count: local_shards.len(),
            total_vectors,
            total_size_bytes,
            total_queries,
            avg_query_time_ms: avg_query_time,
            memory_usage_mb: self.get_memory_usage().await?,
            cpu_usage: self.get_cpu_usage().await?,
        })
    }
    
    async fn get_memory_usage(&self) -> Result<f64> {
        // Placeholder for actual memory usage calculation
        Ok(512.0) // MB
    }
    
    async fn get_cpu_usage(&self) -> Result<f64> {
        // Placeholder for actual CPU usage calculation
        Ok(0.25) // 25%
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStatistics {
    pub node_id: NodeId,
    pub shard_count: usize,
    pub total_vectors: u64,
    pub total_size_bytes: u64,
    pub total_queries: u64,
    pub avg_query_time_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage: f64,
}

impl QueryProcessor {
    pub fn new(config: WorkerConfig, monitoring: Arc<MonitoringSystem>) -> Self {
        Self {
            worker_config: config,
            monitoring,
        }
    }
    
    pub async fn similarity_search(
        &self,
        query_vector: &[f32],
        storage: &VectorStorage,
        index: &LocalIndex,
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        // Use optimized similarity search based on index type
        match index.index_type {
            IndexType::SpiralOptimized => {
                self.spiral_optimized_search(query_vector, storage, k).await
            }
            IndexType::Hnsw => {
                self.hnsw_search(query_vector, storage, k).await
            }
            IndexType::IvfFlat => {
                self.ivf_flat_search(query_vector, storage, k).await
            }
        }
    }
    
    async fn spiral_optimized_search(
        &self,
        query_vector: &[f32],
        storage: &VectorStorage,
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        // Implement spiral-delta optimized search
        let mut results = Vec::new();
        
        for (idx, vector) in storage.vectors.iter().enumerate() {
            let similarity = cosine_similarity(query_vector, vector);
            results.push((idx, similarity));
        }
        
        // Sort by similarity (descending) and take top-k
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        
        Ok(results)
    }
    
    async fn hnsw_search(
        &self,
        _query_vector: &[f32],
        _storage: &VectorStorage,
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        // Placeholder for HNSW search implementation
        Ok((0..k).map(|i| (i, 1.0 - i as f32 * 0.1)).collect())
    }
    
    async fn ivf_flat_search(
        &self,
        _query_vector: &[f32],
        _storage: &VectorStorage,
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        // Placeholder for IVF-Flat search implementation
        Ok((0..k).map(|i| (i, 1.0 - i as f32 * 0.1)).collect())
    }
    
    pub async fn build_index(&self, storage: &VectorStorage) -> Result<LocalIndex> {
        // Build an optimized index for the vectors
        Ok(LocalIndex {
            index_type: IndexType::SpiralOptimized,
            index_data: vec![0; 1024], // Placeholder serialized index
            build_timestamp: chrono::Utc::now(),
        })
    }
    
    pub async fn optimize_index(&self, _shard_id: ShardId, _storage: &VectorStorage) -> Result<()> {
        // Placeholder for index optimization
        Ok(())
    }
    
    pub async fn rebuild_index(&self, _shard_id: ShardId, _storage: &VectorStorage) -> Result<()> {
        // Placeholder for index rebuilding
        Ok(())
    }
}

impl StorageEngine {
    pub async fn new(config: &WorkerConfig) -> Result<Self> {
        let storage_path = std::path::PathBuf::from("./data/worker_storage");
        
        Ok(Self {
            storage_path,
            compression_enabled: config.enable_compression,
            db_handle: None,
        })
    }
    
    pub async fn initialize(&mut self) -> Result<()> {
        // Create storage directory
        std::fs::create_dir_all(&self.storage_path)?;
        
        // Initialize RocksDB
        let db_path = self.storage_path.join("rocksdb");
        let db = rocksdb::DB::open_default(db_path)?;
        self.db_handle = Some(db);
        
        info!("Storage engine initialized at {:?}", self.storage_path);
        Ok(())
    }
    
    pub async fn list_shards(&self) -> Result<Vec<ShardId>> {
        // Placeholder - would scan storage for existing shards
        Ok(vec![])
    }
    
    pub async fn load_shard(&self, _shard_id: ShardId) -> Result<LocalShard> {
        // Placeholder for loading shard from storage
        Ok(LocalShard {
            shard_info: ShardInfo {
                shard_id: _shard_id,
                hash_range: (0, u64::MAX),
                primary_node: "worker-1".to_string(),
                replica_nodes: vec![],
                size_bytes: 0,
                vector_count: 0,
                status: ShardStatus::Active,
                last_updated: chrono::Utc::now(),
            },
            vector_storage: VectorStorage {
                vectors: vec![],
                metadata: vec![],
                size_bytes: 0,
            },
            index: LocalIndex {
                index_type: IndexType::SpiralOptimized,
                index_data: vec![],
                build_timestamp: chrono::Utc::now(),
            },
            statistics: ShardStatistics::default(),
        })
    }
    
    pub async fn persist_vector(&self, _shard_id: ShardId, _vector_id: u64, _vector: &[f32]) -> Result<()> {
        // Placeholder for vector persistence
        Ok(())
    }
}

// gRPC service implementation
struct WorkerService {
    local_shards: Arc<RwLock<HashMap<ShardId, LocalShard>>>,
    query_processor: Arc<QueryProcessor>,
    monitoring: Arc<MonitoringSystem>,
}

impl WorkerService {
    fn new(
        local_shards: Arc<RwLock<HashMap<ShardId, LocalShard>>>,
        query_processor: Arc<QueryProcessor>,
        monitoring: Arc<MonitoringSystem>,
    ) -> Self {
        Self {
            local_shards,
            query_processor,
            monitoring,
        }
    }
    
    fn into_service(self) -> impl tonic::server::NamedService {
        // Placeholder - would implement actual gRPC service
        todo!("Implement gRPC service traits")
    }
}

// Utility function for cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            max_shard_memory_gb: 4.0,
            max_concurrent_queries: 100,
            query_timeout_ms: 30000,
            enable_compression: true,
            replication: ReplicationConfig {
                enabled: true,
                factor: 2,
                sync_mode: ReplicationSyncMode::Asynchronous,
            },
        }
    }
}