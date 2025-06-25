/*!
Data Partitioning and Sharding

Implements consistent hashing and data partitioning strategies for distributed storage.
*/

use crate::*;
use anyhow::{Result, anyhow};
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug, instrument};
use serde::{Serialize, Deserialize};
use blake3::Hasher;

/// Partitioning manager for distributed data
pub struct PartitioningManager {
    /// Partitioning configuration
    config: PartitioningConfig,
    
    /// Current partitioning scheme
    scheme: Arc<RwLock<PartitioningScheme>>,
    
    /// Shard assignments
    shard_assignments: Arc<RwLock<HashMap<ShardId, ShardAssignment>>>,
    
    /// Rebalancing state
    rebalancing_state: Arc<RwLock<Option<RebalancingState>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitioningConfig {
    /// Number of virtual nodes per physical node for consistent hashing
    pub virtual_nodes_per_node: usize,
    
    /// Target number of shards
    pub target_shard_count: usize,
    
    /// Maximum shard size in bytes
    pub max_shard_size_bytes: u64,
    
    /// Replication factor
    pub replication_factor: usize,
    
    /// Partitioning strategy
    pub strategy: PartitioningStrategy,
    
    /// Rebalancing configuration
    pub rebalancing: RebalancingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitioningStrategy {
    /// Consistent hashing with virtual nodes
    ConsistentHashing,
    
    /// Range-based partitioning
    RangeBased,
    
    /// Hash-based partitioning
    HashBased,
    
    /// Custom spiral-delta optimized partitioning
    SpiralOptimized,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancingConfig {
    /// Enable automatic rebalancing
    pub auto_rebalancing: bool,
    
    /// Threshold for triggering rebalancing (load imbalance ratio)
    pub load_threshold: f64,
    
    /// Maximum concurrent rebalancing operations
    pub max_concurrent_operations: usize,
    
    /// Rebalancing schedule (in seconds)
    pub check_interval_seconds: u64,
}

/// Partitioning scheme defining how data is distributed
#[derive(Debug, Clone)]
pub struct PartitioningScheme {
    /// Partitioning strategy
    strategy: PartitioningStrategy,
    
    /// Hash ring for consistent hashing
    hash_ring: ConsistentHashRing,
    
    /// Range assignments for range-based partitioning
    range_assignments: BTreeMap<u64, NodeId>,
    
    /// Shard to node mapping
    shard_to_nodes: HashMap<ShardId, Vec<NodeId>>,
    
    /// Node to shards mapping
    node_to_shards: HashMap<NodeId, Vec<ShardId>>,
}

/// Consistent hash ring implementation
#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    /// Ring positions mapped to nodes
    ring: BTreeMap<u64, NodeId>,
    
    /// Virtual node count per physical node
    virtual_nodes: usize,
    
    /// Active nodes in the ring
    nodes: HashMap<NodeId, NodeInfo>,
}

#[derive(Debug, Clone)]
pub struct ShardAssignment {
    /// Shard identifier
    pub shard_id: ShardId,
    
    /// Primary node
    pub primary_node: NodeId,
    
    /// Replica nodes
    pub replica_nodes: Vec<NodeId>,
    
    /// Hash range covered by this shard
    pub hash_range: (u64, u64),
    
    /// Current shard size
    pub size_bytes: u64,
    
    /// Vector count in shard
    pub vector_count: u64,
    
    /// Assignment timestamp
    pub assigned_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug)]
struct RebalancingState {
    /// Operations in progress
    operations: Vec<RebalancingOperation>,
    
    /// Start time of rebalancing
    started_at: chrono::DateTime<chrono::Utc>,
    
    /// Target completion time
    target_completion: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
struct RebalancingOperation {
    /// Operation identifier
    id: String,
    
    /// Operation type
    operation_type: RebalancingOperationType,
    
    /// Source node
    source_node: NodeId,
    
    /// Target node
    target_node: NodeId,
    
    /// Shard being moved
    shard_id: ShardId,
    
    /// Operation status
    status: RebalancingStatus,
    
    /// Progress (0.0 to 1.0)
    progress: f64,
    
    /// Started timestamp
    started_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
enum RebalancingOperationType {
    /// Move shard between nodes
    ShardMigration,
    
    /// Split oversized shard
    ShardSplit,
    
    /// Merge undersized shards
    ShardMerge,
    
    /// Replicate shard to new node
    ShardReplication,
}

#[derive(Debug, Clone)]
enum RebalancingStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

impl PartitioningManager {
    /// Create a new partitioning manager
    pub async fn new(config: PartitioningConfig) -> Result<Self> {
        let scheme = Arc::new(RwLock::new(PartitioningScheme::new(config.clone())?));
        
        Ok(Self {
            config,
            scheme,
            shard_assignments: Arc::new(RwLock::new(HashMap::new())),
            rebalancing_state: Arc::new(RwLock::new(None)),
        })
    }
    
    /// Add a node to the partitioning scheme
    #[instrument(skip(self))]
    pub async fn add_node(&self, node_info: NodeInfo) -> Result<()> {
        let mut scheme = self.scheme.write().await;
        scheme.add_node(node_info.clone())?;
        
        info!("Added node {} to partitioning scheme", node_info.node_id);
        
        // Trigger rebalancing if auto-rebalancing is enabled
        if self.config.rebalancing.auto_rebalancing {
            drop(scheme);
            self.trigger_rebalancing().await?;
        }
        
        Ok(())
    }
    
    /// Remove a node from the partitioning scheme
    #[instrument(skip(self))]
    pub async fn remove_node(&self, node_id: &NodeId) -> Result<()> {
        let mut scheme = self.scheme.write().await;
        scheme.remove_node(node_id)?;
        
        info!("Removed node {} from partitioning scheme", node_id);
        
        // Trigger rebalancing to redistribute data
        if self.config.rebalancing.auto_rebalancing {
            drop(scheme);
            self.trigger_rebalancing().await?;
        }
        
        Ok(())
    }
    
    /// Get the primary node for a given key
    #[instrument(skip(self))]
    pub async fn get_node_for_key(&self, key: &str) -> Result<NodeId> {
        let scheme = self.scheme.read().await;
        scheme.get_node_for_key(key)
    }
    
    /// Get nodes for storing a vector (including replicas)
    #[instrument(skip(self))]
    pub async fn get_nodes_for_vector(&self, vector: &[f32]) -> Result<Vec<NodeId>> {
        let key = self.vector_to_key(vector)?;
        let scheme = self.scheme.read().await;
        
        let primary_node = scheme.get_node_for_key(&key)?;
        let replica_nodes = scheme.get_replica_nodes(&primary_node, self.config.replication_factor)?;
        
        let mut all_nodes = vec![primary_node];
        all_nodes.extend(replica_nodes);
        
        Ok(all_nodes)
    }
    
    /// Get shard assignments for a node
    pub async fn get_shards_for_node(&self, node_id: &NodeId) -> Result<Vec<ShardAssignment>> {
        let assignments = self.shard_assignments.read().await;
        let node_shards: Vec<ShardAssignment> = assignments.values()
            .filter(|assignment| {
                assignment.primary_node == *node_id || 
                assignment.replica_nodes.contains(node_id)
            })
            .cloned()
            .collect();
        
        Ok(node_shards)
    }
    
    /// Create a new shard assignment
    #[instrument(skip(self))]
    pub async fn create_shard_assignment(
        &self,
        hash_range: (u64, u64),
        primary_node: NodeId,
        replica_nodes: Vec<NodeId>,
    ) -> Result<ShardId> {
        let shard_id = self.generate_shard_id().await;
        
        let assignment = ShardAssignment {
            shard_id,
            primary_node,
            replica_nodes,
            hash_range,
            size_bytes: 0,
            vector_count: 0,
            assigned_at: chrono::Utc::now(),
        };
        
        let mut assignments = self.shard_assignments.write().await;
        assignments.insert(shard_id, assignment);
        
        info!("Created shard assignment {} for range {:?}", shard_id, hash_range);
        Ok(shard_id)
    }
    
    /// Update shard statistics
    pub async fn update_shard_stats(&self, shard_id: ShardId, size_bytes: u64, vector_count: u64) -> Result<()> {
        let mut assignments = self.shard_assignments.write().await;
        
        if let Some(assignment) = assignments.get_mut(&shard_id) {
            assignment.size_bytes = size_bytes;
            assignment.vector_count = vector_count;
            
            // Check if shard needs splitting
            if size_bytes > self.config.max_shard_size_bytes {
                info!("Shard {} exceeds size limit, marking for split", shard_id);
                // In a real implementation, this would trigger shard splitting
            }
        }
        
        Ok(())
    }
    
    /// Trigger rebalancing operation
    async fn trigger_rebalancing(&self) -> Result<()> {
        let mut rebalancing_state = self.rebalancing_state.write().await;
        
        if rebalancing_state.is_some() {
            return Ok(()); // Rebalancing already in progress
        }
        
        info!("Starting cluster rebalancing");
        
        // Analyze current load distribution
        let operations = self.plan_rebalancing_operations().await?;
        
        if operations.is_empty() {
            info!("No rebalancing operations needed");
            return Ok(());
        }
        
        *rebalancing_state = Some(RebalancingState {
            operations,
            started_at: chrono::Utc::now(),
            target_completion: chrono::Utc::now() + chrono::Duration::hours(1),
        });
        
        drop(rebalancing_state);
        
        // Start rebalancing in background
        let manager = self.clone();
        tokio::spawn(async move {
            if let Err(e) = manager.execute_rebalancing().await {
                warn!("Rebalancing failed: {}", e);
            }
        });
        
        Ok(())
    }
    
    async fn plan_rebalancing_operations(&self) -> Result<Vec<RebalancingOperation>> {
        let assignments = self.shard_assignments.read().await;
        let scheme = self.scheme.read().await;
        
        let mut operations = Vec::new();
        
        // Calculate load per node
        let mut node_loads: HashMap<NodeId, (u64, usize)> = HashMap::new(); // (size_bytes, shard_count)
        
        for assignment in assignments.values() {
            let entry = node_loads.entry(assignment.primary_node.clone()).or_default();
            entry.0 += assignment.size_bytes;
            entry.1 += 1;
        }
        
        // Find overloaded and underloaded nodes
        let total_nodes = scheme.hash_ring.nodes.len();
        if total_nodes == 0 {
            return Ok(operations);
        }
        
        let total_size: u64 = node_loads.values().map(|(size, _)| *size).sum();
        let avg_size_per_node = total_size / total_nodes as u64;
        
        let mut overloaded_nodes = Vec::new();
        let mut underloaded_nodes = Vec::new();
        
        for (node_id, &(size, _count)) in &node_loads {
            let load_ratio = size as f64 / avg_size_per_node as f64;
            
            if load_ratio > 1.0 + self.config.rebalancing.load_threshold {
                overloaded_nodes.push((node_id.clone(), size));
            } else if load_ratio < 1.0 - self.config.rebalancing.load_threshold {
                underloaded_nodes.push((node_id.clone(), size));
            }
        }
        
        // Plan migrations from overloaded to underloaded nodes
        for (overloaded_node, _) in overloaded_nodes {
            if let Some((underloaded_node, _)) = underloaded_nodes.first() {
                // Find a shard to migrate
                let shard_to_migrate = assignments.values()
                    .filter(|a| a.primary_node == overloaded_node)
                    .min_by_key(|a| a.size_bytes); // Choose smallest shard first
                
                if let Some(shard) = shard_to_migrate {
                    operations.push(RebalancingOperation {
                        id: uuid::Uuid::new_v4().to_string(),
                        operation_type: RebalancingOperationType::ShardMigration,
                        source_node: overloaded_node.clone(),
                        target_node: underloaded_node.clone(),
                        shard_id: shard.shard_id,
                        status: RebalancingStatus::Pending,
                        progress: 0.0,
                        started_at: chrono::Utc::now(),
                    });
                }
            }
        }
        
        Ok(operations)
    }
    
    async fn execute_rebalancing(&self) -> Result<()> {
        loop {
            let mut rebalancing_state = self.rebalancing_state.write().await;
            
            let state = match rebalancing_state.as_mut() {
                Some(state) => state,
                None => break, // No rebalancing in progress
            };
            
            // Find next pending operation
            let next_operation = state.operations.iter_mut()
                .find(|op| matches!(op.status, RebalancingStatus::Pending));
            
            if let Some(operation) = next_operation {
                operation.status = RebalancingStatus::InProgress;
                let op_clone = operation.clone();
                drop(rebalancing_state);
                
                // Execute the operation
                match self.execute_rebalancing_operation(&op_clone).await {
                    Ok(_) => {
                        let mut state = self.rebalancing_state.write().await;
                        if let Some(state) = state.as_mut() {
                            if let Some(op) = state.operations.iter_mut().find(|o| o.id == op_clone.id) {
                                op.status = RebalancingStatus::Completed;
                                op.progress = 1.0;
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Rebalancing operation {} failed: {}", op_clone.id, e);
                        let mut state = self.rebalancing_state.write().await;
                        if let Some(state) = state.as_mut() {
                            if let Some(op) = state.operations.iter_mut().find(|o| o.id == op_clone.id) {
                                op.status = RebalancingStatus::Failed;
                            }
                        }
                    }
                }
            } else {
                // All operations completed
                let all_completed = state.operations.iter()
                    .all(|op| matches!(op.status, RebalancingStatus::Completed | RebalancingStatus::Failed));
                
                if all_completed {
                    info!("Rebalancing completed");
                    *rebalancing_state = None;
                    break;
                }
            }
            
            drop(rebalancing_state);
            
            // Wait a bit before checking again
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        }
        
        Ok(())
    }
    
    async fn execute_rebalancing_operation(&self, operation: &RebalancingOperation) -> Result<()> {
        match operation.operation_type {
            RebalancingOperationType::ShardMigration => {
                self.migrate_shard(operation.shard_id, &operation.source_node, &operation.target_node).await
            }
            RebalancingOperationType::ShardSplit => {
                self.split_shard(operation.shard_id).await
            }
            RebalancingOperationType::ShardMerge => {
                // Placeholder - would merge multiple shards
                Ok(())
            }
            RebalancingOperationType::ShardReplication => {
                self.replicate_shard(operation.shard_id, &operation.target_node).await
            }
        }
    }
    
    async fn migrate_shard(&self, shard_id: ShardId, source_node: &NodeId, target_node: &NodeId) -> Result<()> {
        info!("Migrating shard {} from {} to {}", shard_id, source_node, target_node);
        
        // Update shard assignment
        let mut assignments = self.shard_assignments.write().await;
        if let Some(assignment) = assignments.get_mut(&shard_id) {
            assignment.primary_node = target_node.clone();
        }
        
        // In a real implementation, this would:
        // 1. Copy shard data to target node
        // 2. Verify data integrity
        // 3. Update routing tables
        // 4. Remove data from source node
        
        Ok(())
    }
    
    async fn split_shard(&self, shard_id: ShardId) -> Result<()> {
        info!("Splitting shard {}", shard_id);
        
        // In a real implementation, this would:
        // 1. Create two new shards
        // 2. Redistribute data between them
        // 3. Update hash ranges
        // 4. Remove original shard
        
        Ok(())
    }
    
    async fn replicate_shard(&self, shard_id: ShardId, target_node: &NodeId) -> Result<()> {
        info!("Replicating shard {} to {}", shard_id, target_node);
        
        // Update shard assignment to include new replica
        let mut assignments = self.shard_assignments.write().await;
        if let Some(assignment) = assignments.get_mut(&shard_id) {
            if !assignment.replica_nodes.contains(target_node) {
                assignment.replica_nodes.push(target_node.clone());
            }
        }
        
        Ok(())
    }
    
    fn vector_to_key(&self, vector: &[f32]) -> Result<String> {
        // Create a deterministic key from vector
        let mut hasher = Hasher::new();
        
        for &component in vector {
            hasher.update(&component.to_le_bytes());
        }
        
        Ok(hasher.finalize().to_hex().to_string())
    }
    
    async fn generate_shard_id(&self) -> ShardId {
        // Simple shard ID generation
        let assignments = self.shard_assignments.read().await;
        assignments.len() as u64 + 1
    }
    
    /// Get rebalancing status
    pub async fn get_rebalancing_status(&self) -> Option<RebalancingStatus> {
        let state = self.rebalancing_state.read().await;
        
        if let Some(state) = state.as_ref() {
            if state.operations.iter().any(|op| matches!(op.status, RebalancingStatus::InProgress)) {
                Some(RebalancingStatus::InProgress)
            } else if state.operations.iter().all(|op| matches!(op.status, RebalancingStatus::Completed)) {
                Some(RebalancingStatus::Completed)
            } else {
                Some(RebalancingStatus::Pending)
            }
        } else {
            None
        }
    }
}

impl PartitioningScheme {
    fn new(config: PartitioningConfig) -> Result<Self> {
        let hash_ring = ConsistentHashRing::new(config.virtual_nodes_per_node);
        
        Ok(Self {
            strategy: config.strategy,
            hash_ring,
            range_assignments: BTreeMap::new(),
            shard_to_nodes: HashMap::new(),
            node_to_shards: HashMap::new(),
        })
    }
    
    fn add_node(&mut self, node_info: NodeInfo) -> Result<()> {
        self.hash_ring.add_node(node_info.clone())?;
        self.node_to_shards.insert(node_info.node_id.clone(), Vec::new());
        Ok(())
    }
    
    fn remove_node(&mut self, node_id: &NodeId) -> Result<()> {
        self.hash_ring.remove_node(node_id)?;
        self.node_to_shards.remove(node_id);
        
        // Remove node from shard assignments
        for nodes in self.shard_to_nodes.values_mut() {
            nodes.retain(|id| id != node_id);
        }
        
        Ok(())
    }
    
    fn get_node_for_key(&self, key: &str) -> Result<NodeId> {
        match self.strategy {
            PartitioningStrategy::ConsistentHashing | 
            PartitioningStrategy::SpiralOptimized => {
                self.hash_ring.get_node(key)
            }
            PartitioningStrategy::HashBased => {
                let hash = self.hash_key(key);
                self.hash_ring.get_node_by_hash(hash)
            }
            PartitioningStrategy::RangeBased => {
                let hash = self.hash_key(key);
                self.range_assignments.range(hash..)
                    .next()
                    .map(|(_, node_id)| node_id.clone())
                    .ok_or_else(|| anyhow!("No node found for key"))
            }
        }
    }
    
    fn get_replica_nodes(&self, primary_node: &NodeId, replication_factor: usize) -> Result<Vec<NodeId>> {
        let mut replicas = Vec::new();
        let all_nodes: Vec<_> = self.hash_ring.nodes.keys().cloned().collect();
        
        // Simple replica selection - choose next nodes in consistent order
        if let Some(primary_index) = all_nodes.iter().position(|id| id == primary_node) {
            for i in 1..replication_factor {
                let replica_index = (primary_index + i) % all_nodes.len();
                if replica_index != primary_index {
                    replicas.push(all_nodes[replica_index].clone());
                }
            }
        }
        
        Ok(replicas)
    }
    
    fn hash_key(&self, key: &str) -> u64 {
        let mut hasher = Hasher::new();
        hasher.update(key.as_bytes());
        let hash = hasher.finalize();
        u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap_or([0; 8]))
    }
}

impl ConsistentHashRing {
    fn new(virtual_nodes: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            virtual_nodes,
            nodes: HashMap::new(),
        }
    }
    
    fn add_node(&mut self, node_info: NodeInfo) -> Result<()> {
        let node_id = node_info.node_id.clone();
        
        // Add virtual nodes to the ring
        for i in 0..self.virtual_nodes {
            let virtual_key = format!("{}:{}", node_id, i);
            let hash = self.hash_string(&virtual_key);
            self.ring.insert(hash, node_id.clone());
        }
        
        self.nodes.insert(node_id, node_info);
        Ok(())
    }
    
    fn remove_node(&mut self, node_id: &NodeId) -> Result<()> {
        // Remove virtual nodes from the ring
        let mut keys_to_remove = Vec::new();
        for (&hash, id) in &self.ring {
            if id == node_id {
                keys_to_remove.push(hash);
            }
        }
        
        for key in keys_to_remove {
            self.ring.remove(&key);
        }
        
        self.nodes.remove(node_id);
        Ok(())
    }
    
    fn get_node(&self, key: &str) -> Result<NodeId> {
        let hash = self.hash_string(key);
        self.get_node_by_hash(hash)
    }
    
    fn get_node_by_hash(&self, hash: u64) -> Result<NodeId> {
        if self.ring.is_empty() {
            return Err(anyhow!("No nodes in hash ring"));
        }
        
        // Find the first node >= hash, or wrap around to the first node
        self.ring.range(hash..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, node_id)| node_id.clone())
            .ok_or_else(|| anyhow!("No node found in hash ring"))
    }
    
    fn hash_string(&self, s: &str) -> u64 {
        let mut hasher = Hasher::new();
        hasher.update(s.as_bytes());
        let hash = hasher.finalize();
        u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap_or([0; 8]))
    }
}

impl Clone for PartitioningManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            scheme: self.scheme.clone(),
            shard_assignments: self.shard_assignments.clone(),
            rebalancing_state: self.rebalancing_state.clone(),
        }
    }
}

impl Default for PartitioningConfig {
    fn default() -> Self {
        Self {
            virtual_nodes_per_node: 150,
            target_shard_count: 1000,
            max_shard_size_bytes: 1024 * 1024 * 1024, // 1GB
            replication_factor: 2,
            strategy: PartitioningStrategy::SpiralOptimized,
            rebalancing: RebalancingConfig {
                auto_rebalancing: true,
                load_threshold: 0.2, // 20% imbalance triggers rebalancing
                max_concurrent_operations: 3,
                check_interval_seconds: 300, // 5 minutes
            },
        }
    }
}

use uuid;