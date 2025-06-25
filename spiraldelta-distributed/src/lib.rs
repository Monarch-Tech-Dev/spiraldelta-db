/*!
SpiralDelta Distributed Architecture

This module provides distributed computing capabilities for SpiralDeltaDB:
- Horizontal scaling across multiple nodes
- Consistent hashing for data partitioning
- Raft consensus for coordination
- Load balancing and fault tolerance
- Cross-node query processing

Architecture Components:
- Coordinator: Manages cluster topology and query planning
- Workers: Execute queries and store data shards
- Gateway: Client-facing API with load balancing
*/

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

pub mod coordinator;
pub mod worker;
pub mod gateway;
pub mod consensus;
pub mod partitioning;
pub mod replication;
pub mod query_engine;
pub mod load_balancer;
pub mod monitoring;

pub use coordinator::*;
pub use worker::*;
pub use gateway::*;
pub use consensus::*;
pub use partitioning::*;
pub use replication::*;
pub use query_engine::*;
pub use load_balancer::*;
pub use monitoring::*;

/// Unique identifier for cluster nodes
pub type NodeId = String;

/// Unique identifier for data shards
pub type ShardId = u64;

/// Unique identifier for queries
pub type QueryId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Cluster name for identification
    pub cluster_name: String,
    
    /// Replication factor for data
    pub replication_factor: usize,
    
    /// Number of virtual nodes for consistent hashing
    pub virtual_nodes: usize,
    
    /// Consensus configuration
    pub consensus: ConsensusConfig,
    
    /// Load balancer settings
    pub load_balancer: LoadBalancerConfig,
    
    /// Monitoring and metrics
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Raft election timeout in milliseconds
    pub election_timeout_ms: u64,
    
    /// Raft heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    
    /// Maximum log entries per append
    pub max_append_entries: usize,
    
    /// Snapshot threshold (log entries)
    pub snapshot_threshold: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    
    /// Health check interval in seconds
    pub health_check_interval_s: u64,
    
    /// Maximum concurrent requests per node
    pub max_concurrent_requests: usize,
    
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    ConsistentHashing,
    LoadAware,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable Prometheus metrics
    pub enable_metrics: bool,
    
    /// Metrics port
    pub metrics_port: u16,
    
    /// Enable distributed tracing
    pub enable_tracing: bool,
    
    /// Log level
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Unique node identifier
    pub node_id: NodeId,
    
    /// Node address
    pub address: SocketAddr,
    
    /// Node type (coordinator, worker, gateway)
    pub node_type: NodeType,
    
    /// Node status
    pub status: NodeStatus,
    
    /// Node capabilities
    pub capabilities: NodeCapabilities,
    
    /// Resource information
    pub resources: NodeResources,
    
    /// Last heartbeat timestamp
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    Coordinator,
    Worker,
    Gateway,
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Starting,
    Healthy,
    Degraded,
    Unhealthy,
    Shutting Down,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// Maximum memory available (GB)
    pub max_memory_gb: f64,
    
    /// CPU cores available
    pub cpu_cores: usize,
    
    /// GPU acceleration available
    pub gpu_acceleration: bool,
    
    /// Storage capacity (GB)
    pub storage_capacity_gb: f64,
    
    /// Network bandwidth (Mbps)
    pub network_bandwidth_mbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResources {
    /// Current memory usage (GB)
    pub memory_used_gb: f64,
    
    /// Current CPU usage (0.0-1.0)
    pub cpu_usage: f64,
    
    /// Current storage usage (GB)
    pub storage_used_gb: f64,
    
    /// Active connections count
    pub active_connections: usize,
    
    /// Queries per second
    pub queries_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    /// Unique shard identifier
    pub shard_id: ShardId,
    
    /// Hash range covered by this shard
    pub hash_range: (u64, u64),
    
    /// Primary node for this shard
    pub primary_node: NodeId,
    
    /// Replica nodes for this shard
    pub replica_nodes: Vec<NodeId>,
    
    /// Shard size in bytes
    pub size_bytes: u64,
    
    /// Number of vectors in shard
    pub vector_count: u64,
    
    /// Shard status
    pub status: ShardStatus,
    
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardStatus {
    Initializing,
    Active,
    Migrating,
    ReadOnly,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedQuery {
    /// Unique query identifier
    pub query_id: QueryId,
    
    /// Original query from client
    pub original_query: Vec<f32>,
    
    /// Query type (similarity search, index construction, etc.)
    pub query_type: QueryType,
    
    /// Query parameters
    pub parameters: QueryParameters,
    
    /// Target shards for this query
    pub target_shards: Vec<ShardId>,
    
    /// Query execution plan
    pub execution_plan: ExecutionPlan,
    
    /// Query priority
    pub priority: QueryPriority,
    
    /// Query deadline
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryType {
    SimilaritySearch,
    VectorInsertion,
    IndexConstruction,
    BatchOperations,
    ClusterStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParameters {
    /// Number of nearest neighbors to return
    pub k: Option<usize>,
    
    /// Similarity metric to use
    pub metric: Option<String>,
    
    /// Search radius for range queries
    pub radius: Option<f64>,
    
    /// Additional parameters as key-value pairs
    pub extra_params: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Query execution steps
    pub steps: Vec<ExecutionStep>,
    
    /// Estimated execution time
    pub estimated_duration_ms: u64,
    
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStep {
    /// Step identifier
    pub step_id: String,
    
    /// Target nodes for this step
    pub target_nodes: Vec<NodeId>,
    
    /// Step type
    pub step_type: StepType,
    
    /// Step dependencies
    pub dependencies: Vec<String>,
    
    /// Estimated duration
    pub estimated_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    ShardQuery,
    ResultAggregation,
    DataMigration,
    IndexUpdate,
    Cleanup,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Memory required (GB)
    pub memory_gb: f64,
    
    /// CPU cores required
    pub cpu_cores: f64,
    
    /// Network bandwidth required (Mbps)
    pub network_mbps: f64,
    
    /// GPU acceleration required
    pub gpu_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Query identifier
    pub query_id: QueryId,
    
    /// Execution status
    pub status: QueryStatus,
    
    /// Query results
    pub results: QueryResultData,
    
    /// Execution statistics
    pub statistics: QueryStatistics,
    
    /// Error information if failed
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryResultData {
    SimilarityResults(Vec<(usize, f32)>),
    InsertionResult { vector_id: u64, shard_id: ShardId },
    IndexResult { index_id: String, status: String },
    ClusterInfo(ClusterInfo),
    BatchResults(Vec<QueryResult>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatistics {
    /// Total execution time
    pub total_duration_ms: u64,
    
    /// Number of nodes involved
    pub nodes_involved: usize,
    
    /// Number of shards queried
    pub shards_queried: usize,
    
    /// Network round trips
    pub network_round_trips: usize,
    
    /// Data transferred (bytes)
    pub data_transferred_bytes: u64,
    
    /// Resource usage per node
    pub per_node_stats: HashMap<NodeId, NodeQueryStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeQueryStats {
    /// Execution time on this node
    pub duration_ms: u64,
    
    /// Memory used
    pub memory_used_mb: f64,
    
    /// CPU time used
    pub cpu_time_ms: u64,
    
    /// Network I/O
    pub network_io_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    /// Cluster name
    pub cluster_name: String,
    
    /// Total nodes in cluster
    pub total_nodes: usize,
    
    /// Healthy nodes count
    pub healthy_nodes: usize,
    
    /// Total shards
    pub total_shards: usize,
    
    /// Active shards
    pub active_shards: usize,
    
    /// Total vectors stored
    pub total_vectors: u64,
    
    /// Total storage used (bytes)
    pub total_storage_bytes: u64,
    
    /// Cluster resource utilization
    pub resource_utilization: ClusterResources,
    
    /// Performance metrics
    pub performance_metrics: ClusterPerformance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterResources {
    /// Total memory available (GB)
    pub total_memory_gb: f64,
    
    /// Memory currently used (GB)
    pub used_memory_gb: f64,
    
    /// Total CPU cores
    pub total_cpu_cores: usize,
    
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,
    
    /// Total storage capacity (GB)
    pub total_storage_gb: f64,
    
    /// Storage utilization (GB)
    pub used_storage_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPerformance {
    /// Queries per second (cluster-wide)
    pub queries_per_second: f64,
    
    /// Average query latency (ms)
    pub avg_query_latency_ms: f64,
    
    /// 95th percentile latency (ms)
    pub p95_latency_ms: f64,
    
    /// 99th percentile latency (ms)
    pub p99_latency_ms: f64,
    
    /// Error rate (0.0-1.0)
    pub error_rate: f64,
    
    /// Throughput (vectors/second)
    pub throughput_vectors_per_second: f64,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            cluster_name: "spiraldelta-cluster".to_string(),
            replication_factor: 2,
            virtual_nodes: 64,
            consensus: ConsensusConfig::default(),
            load_balancer: LoadBalancerConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            election_timeout_ms: 150,
            heartbeat_interval_ms: 50,
            max_append_entries: 1000,
            snapshot_threshold: 10000,
        }
    }
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::LoadAware,
            health_check_interval_s: 30,
            max_concurrent_requests: 1000,
            request_timeout_ms: 30000,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_port: 9090,
            enable_tracing: true,
            log_level: "info".to_string(),
        }
    }
}

/// Distributed SpiralDelta cluster manager
pub struct DistributedCluster {
    /// Cluster configuration
    config: ClusterConfig,
    
    /// Current cluster topology
    topology: Arc<RwLock<ClusterTopology>>,
    
    /// Query coordinator
    coordinator: Option<Arc<QueryCoordinator>>,
    
    /// Load balancer
    load_balancer: Arc<LoadBalancer>,
    
    /// Monitoring system
    monitoring: Arc<MonitoringSystem>,
}

#[derive(Debug)]
pub struct ClusterTopology {
    /// All nodes in the cluster
    pub nodes: HashMap<NodeId, NodeInfo>,
    
    /// Shard distribution
    pub shards: HashMap<ShardId, ShardInfo>,
    
    /// Consistent hash ring
    pub hash_ring: consistent_hash::ConsistentHash<NodeId>,
    
    /// Cluster metadata
    pub metadata: ClusterMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterMetadata {
    /// Cluster version
    pub version: String,
    
    /// Configuration checksum
    pub config_checksum: String,
    
    /// Last topology change
    pub last_change: chrono::DateTime<chrono::Utc>,
    
    /// Leader node (for coordination)
    pub leader_node: Option<NodeId>,
}

impl DistributedCluster {
    /// Create a new distributed cluster
    pub async fn new(config: ClusterConfig) -> Result<Self> {
        let topology = Arc::new(RwLock::new(ClusterTopology::new(&config)?));
        let load_balancer = Arc::new(LoadBalancer::new(config.load_balancer.clone()).await?);
        let monitoring = Arc::new(MonitoringSystem::new(config.monitoring.clone()).await?);
        
        Ok(Self {
            config,
            topology,
            coordinator: None,
            load_balancer,
            monitoring,
        })
    }
    
    /// Initialize cluster with coordinator
    pub async fn initialize_coordinator(&mut self, node_info: NodeInfo) -> Result<()> {
        let coordinator = Arc::new(
            QueryCoordinator::new(
                node_info,
                self.config.clone(),
                self.topology.clone(),
                self.monitoring.clone(),
            ).await?
        );
        
        self.coordinator = Some(coordinator);
        info!("Distributed cluster coordinator initialized");
        Ok(())
    }
    
    /// Get cluster information
    pub async fn get_cluster_info(&self) -> Result<ClusterInfo> {
        let topology = self.topology.read().await;
        let healthy_nodes = topology.nodes.values()
            .filter(|node| matches!(node.status, NodeStatus::Healthy))
            .count();
        
        let active_shards = topology.shards.values()
            .filter(|shard| matches!(shard.status, ShardStatus::Active))
            .count();
        
        let total_vectors = topology.shards.values()
            .map(|shard| shard.vector_count)
            .sum();
        
        let total_storage = topology.shards.values()
            .map(|shard| shard.size_bytes)
            .sum();
        
        // Calculate resource utilization
        let (total_memory, used_memory, total_cores, cpu_utilization, total_storage_cap, used_storage_gb) = 
            topology.nodes.values().fold(
                (0.0, 0.0, 0, 0.0, 0.0, 0.0),
                |(tm, um, tc, cu, tsc, usg), node| {
                    (
                        tm + node.capabilities.max_memory_gb,
                        um + node.resources.memory_used_gb,
                        tc + node.capabilities.cpu_cores,
                        cu + node.resources.cpu_usage,
                        tsc + node.capabilities.storage_capacity_gb,
                        usg + node.resources.storage_used_gb,
                    )
                }
            );
        
        let avg_cpu = if topology.nodes.len() > 0 { cpu_utilization / topology.nodes.len() as f64 } else { 0.0 };
        
        // Get performance metrics from monitoring
        let performance_metrics = self.monitoring.get_cluster_performance().await?;
        
        Ok(ClusterInfo {
            cluster_name: self.config.cluster_name.clone(),
            total_nodes: topology.nodes.len(),
            healthy_nodes,
            total_shards: topology.shards.len(),
            active_shards,
            total_vectors,
            total_storage_bytes: total_storage,
            resource_utilization: ClusterResources {
                total_memory_gb: total_memory,
                used_memory_gb: used_memory,
                total_cpu_cores: total_cores,
                cpu_utilization: avg_cpu,
                total_storage_gb: total_storage_cap,
                used_storage_gb,
            },
            performance_metrics,
        })
    }
    
    /// Execute a distributed query
    pub async fn execute_query(&self, query: DistributedQuery) -> Result<QueryResult> {
        if let Some(coordinator) = &self.coordinator {
            coordinator.execute_query(query).await
        } else {
            Err(anyhow!("No coordinator available for query execution"))
        }
    }
    
    /// Add a new node to the cluster
    pub async fn add_node(&self, node_info: NodeInfo) -> Result<()> {
        let mut topology = self.topology.write().await;
        topology.add_node(node_info)?;
        self.monitoring.track_topology_change("node_added").await?;
        Ok(())
    }
    
    /// Remove a node from the cluster
    pub async fn remove_node(&self, node_id: &NodeId) -> Result<()> {
        let mut topology = self.topology.write().await;
        topology.remove_node(node_id)?;
        self.monitoring.track_topology_change("node_removed").await?;
        Ok(())
    }
    
    /// Get cluster statistics
    pub async fn get_statistics(&self) -> Result<HashMap<String, serde_json::Value>> {
        self.monitoring.get_cluster_statistics().await
    }
}

impl ClusterTopology {
    pub fn new(config: &ClusterConfig) -> Result<Self> {
        let hash_ring = consistent_hash::ConsistentHash::new();
        
        Ok(Self {
            nodes: HashMap::new(),
            shards: HashMap::new(),
            hash_ring,
            metadata: ClusterMetadata {
                version: "1.0.0".to_string(),
                config_checksum: blake3::hash(
                    serde_json::to_string(config)?.as_bytes()
                ).to_hex().to_string(),
                last_change: chrono::Utc::now(),
                leader_node: None,
            },
        })
    }
    
    pub fn add_node(&mut self, node_info: NodeInfo) -> Result<()> {
        let node_id = node_info.node_id.clone();
        
        // Add to hash ring for consistent hashing
        self.hash_ring.add(&node_id, 1);
        
        // Add to nodes map
        self.nodes.insert(node_id.clone(), node_info);
        
        // Update metadata
        self.metadata.last_change = chrono::Utc::now();
        
        info!("Added node {} to cluster topology", node_id);
        Ok(())
    }
    
    pub fn remove_node(&mut self, node_id: &NodeId) -> Result<()> {
        // Remove from hash ring
        self.hash_ring.remove(node_id);
        
        // Remove from nodes map
        if self.nodes.remove(node_id).is_none() {
            return Err(anyhow!("Node {} not found in cluster", node_id));
        }
        
        // Update metadata
        self.metadata.last_change = chrono::Utc::now();
        
        // If this was the leader, clear it
        if self.metadata.leader_node.as_ref() == Some(node_id) {
            self.metadata.leader_node = None;
        }
        
        info!("Removed node {} from cluster topology", node_id);
        Ok(())
    }
    
    pub fn get_node_for_key(&self, key: &str) -> Option<&NodeId> {
        self.hash_ring.get(key)
    }
    
    pub fn get_healthy_nodes(&self) -> Vec<&NodeInfo> {
        self.nodes.values()
            .filter(|node| matches!(node.status, NodeStatus::Healthy))
            .collect()
    }
}