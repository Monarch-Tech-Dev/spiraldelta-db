/*!
Distributed Query Execution Engine

Handles the actual execution of queries across distributed nodes and shards.
*/

use crate::*;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug, instrument};
use serde::{Serialize, Deserialize};
use futures::future::join_all;

/// Query planner that generates execution plans
pub struct QueryPlanner {
    config: ClusterConfig,
    topology: Arc<RwLock<ClusterTopology>>,
}

/// Distributed execution engine
pub struct DistributedExecutionEngine {
    topology: Arc<RwLock<ClusterTopology>>,
    node_clients: Arc<RwLock<HashMap<NodeId, Arc<NodeClient>>>>,
    monitoring: Arc<MonitoringSystem>,
}

/// Result aggregator for combining distributed results
pub struct ResultAggregator {
    // Aggregation strategies for different query types
}

/// Client for communicating with individual nodes
pub struct NodeClient {
    node_id: NodeId,
    address: std::net::SocketAddr,
    client: Option<tonic::transport::Channel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardQueryRequest {
    pub query_id: QueryId,
    pub shard_id: ShardId,
    pub query_vector: Vec<f32>,
    pub query_type: QueryType,
    pub parameters: QueryParameters,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardQueryResponse {
    pub query_id: QueryId,
    pub shard_id: ShardId,
    pub results: Vec<(usize, f32)>,
    pub status: String,
    pub execution_time_ms: u64,
    pub error: Option<String>,
}

impl QueryPlanner {
    pub fn new(config: ClusterConfig, topology: Arc<RwLock<ClusterTopology>>) -> Self {
        Self {
            config,
            topology,
        }
    }
    
    /// Generate an execution plan for a distributed query
    #[instrument(skip(self, query))]
    pub async fn plan_query(&self, query: &DistributedQuery) -> Result<ExecutionPlan> {
        debug!("Planning execution for query type: {:?}", query.query_type);
        
        match query.query_type {
            QueryType::SimilaritySearch => self.plan_similarity_search(query).await,
            QueryType::VectorInsertion => self.plan_vector_insertion(query).await,
            QueryType::IndexConstruction => self.plan_index_construction(query).await,
            QueryType::BatchOperations => self.plan_batch_operations(query).await,
            QueryType::ClusterStatus => self.plan_cluster_status(query).await,
        }
    }
    
    async fn plan_similarity_search(&self, query: &DistributedQuery) -> Result<ExecutionPlan> {
        let topology = self.topology.read().await;
        
        // Determine which shards need to be queried
        let target_shards = if query.target_shards.is_empty() {
            // Query all active shards
            topology.shards.keys()
                .filter(|&&shard_id| {
                    topology.shards.get(&shard_id)
                        .map_or(false, |shard| matches!(shard.status, ShardStatus::Active))
                })
                .copied()
                .collect()
        } else {
            query.target_shards.clone()
        };
        
        let mut steps = Vec::new();
        let mut target_nodes = Vec::new();
        
        // Step 1: Query all relevant shards in parallel
        for &shard_id in &target_shards {
            if let Some(shard) = topology.shards.get(&shard_id) {
                let step_id = format!("shard_query_{}", shard_id);
                let shard_nodes = vec![shard.primary_node.clone()];
                target_nodes.extend(shard_nodes.clone());
                
                steps.push(ExecutionStep {
                    step_id,
                    target_nodes: shard_nodes,
                    step_type: StepType::ShardQuery,
                    dependencies: vec![],
                    estimated_duration_ms: self.estimate_shard_query_time(&shard),
                });
            }
        }
        
        // Step 2: Aggregate results from all shards
        if steps.len() > 1 {
            let aggregation_dependencies: Vec<String> = steps.iter()
                .map(|step| step.step_id.clone())
                .collect();
            
            steps.push(ExecutionStep {
                step_id: "result_aggregation".to_string(),
                target_nodes: vec![], // Coordinator handles aggregation
                step_type: StepType::ResultAggregation,
                dependencies: aggregation_dependencies,
                estimated_duration_ms: 50, // Fast aggregation
            });
        }
        
        // Step 3: Cleanup temporary resources
        steps.push(ExecutionStep {
            step_id: "cleanup".to_string(),
            target_nodes: vec![],
            step_type: StepType::Cleanup,
            dependencies: if steps.len() > 1 { vec!["result_aggregation".to_string()] } else { vec![] },
            estimated_duration_ms: 10,
        });
        
        let total_duration = steps.iter().map(|step| step.estimated_duration_ms).max().unwrap_or(100);
        
        Ok(ExecutionPlan {
            steps,
            estimated_duration_ms: total_duration,
            resource_requirements: ResourceRequirements {
                memory_gb: 0.5, // Minimal memory for coordination
                cpu_cores: 0.1,
                network_mbps: target_shards.len() as f64 * 10.0, // 10 Mbps per shard
                gpu_required: false,
            },
        })
    }
    
    async fn plan_vector_insertion(&self, query: &DistributedQuery) -> Result<ExecutionPlan> {
        let topology = self.topology.read().await;
        
        // Determine target shard based on hash of vector
        let vector_hash = self.hash_vector(&query.original_query);
        let target_node = topology.get_node_for_key(&vector_hash.to_string())
            .ok_or_else(|| anyhow!("No available node for vector insertion"))?;
        
        // Find shard on target node
        let target_shard = topology.shards.values()
            .find(|shard| shard.primary_node == *target_node)
            .ok_or_else(|| anyhow!("No shard found on target node"))?;
        
        let steps = vec![
            ExecutionStep {
                step_id: "vector_insertion".to_string(),
                target_nodes: vec![target_node.clone()],
                step_type: StepType::ShardQuery,
                dependencies: vec![],
                estimated_duration_ms: 100,
            },
            ExecutionStep {
                step_id: "index_update".to_string(),
                target_nodes: vec![target_node.clone()],
                step_type: StepType::IndexUpdate,
                dependencies: vec!["vector_insertion".to_string()],
                estimated_duration_ms: 200,
            },
        ];
        
        Ok(ExecutionPlan {
            steps,
            estimated_duration_ms: 300,
            resource_requirements: ResourceRequirements {
                memory_gb: 0.1,
                cpu_cores: 0.2,
                network_mbps: 5.0,
                gpu_required: false,
            },
        })
    }
    
    async fn plan_index_construction(&self, _query: &DistributedQuery) -> Result<ExecutionPlan> {
        // Placeholder for index construction planning
        Ok(ExecutionPlan {
            steps: vec![],
            estimated_duration_ms: 1000,
            resource_requirements: ResourceRequirements {
                memory_gb: 2.0,
                cpu_cores: 2.0,
                network_mbps: 100.0,
                gpu_required: true,
            },
        })
    }
    
    async fn plan_batch_operations(&self, _query: &DistributedQuery) -> Result<ExecutionPlan> {
        // Placeholder for batch operations planning
        Ok(ExecutionPlan {
            steps: vec![],
            estimated_duration_ms: 500,
            resource_requirements: ResourceRequirements {
                memory_gb: 1.0,
                cpu_cores: 1.0,
                network_mbps: 50.0,
                gpu_required: false,
            },
        })
    }
    
    async fn plan_cluster_status(&self, _query: &DistributedQuery) -> Result<ExecutionPlan> {
        let topology = self.topology.read().await;
        let all_nodes: Vec<NodeId> = topology.nodes.keys().cloned().collect();
        
        let steps = vec![
            ExecutionStep {
                step_id: "collect_node_status".to_string(),
                target_nodes: all_nodes,
                step_type: StepType::ShardQuery,
                dependencies: vec![],
                estimated_duration_ms: 100,
            },
            ExecutionStep {
                step_id: "aggregate_cluster_status".to_string(),
                target_nodes: vec![],
                step_type: StepType::ResultAggregation,
                dependencies: vec!["collect_node_status".to_string()],
                estimated_duration_ms: 50,
            },
        ];
        
        Ok(ExecutionPlan {
            steps,
            estimated_duration_ms: 150,
            resource_requirements: ResourceRequirements {
                memory_gb: 0.1,
                cpu_cores: 0.1,
                network_mbps: 10.0,
                gpu_required: false,
            },
        })
    }
    
    fn estimate_shard_query_time(&self, shard: &ShardInfo) -> u64 {
        // Estimate based on shard size and typical query performance
        let base_time = 100u64; // Base query time in ms
        let size_factor = (shard.vector_count as f64 / 10000.0).log2().max(1.0);
        (base_time as f64 * size_factor) as u64
    }
    
    fn hash_vector(&self, vector: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for &component in vector {
            component.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }
}

impl DistributedExecutionEngine {
    pub async fn new(
        topology: Arc<RwLock<ClusterTopology>>,
        monitoring: Arc<MonitoringSystem>,
    ) -> Result<Self> {
        Ok(Self {
            topology,
            node_clients: Arc::new(RwLock::new(HashMap::new())),
            monitoring,
        })
    }
    
    /// Execute query on specified shards and nodes
    #[instrument(skip(self, query))]
    pub async fn execute_on_shards(
        &self,
        shard_ids: &[ShardId],
        target_nodes: &[NodeId],
        query: DistributedQuery,
    ) -> Result<Vec<ShardQueryResponse>> {
        debug!("Executing query on {} shards across {} nodes", shard_ids.len(), target_nodes.len());
        
        // Create query tasks for each shard
        let mut query_tasks = Vec::new();
        
        for &shard_id in shard_ids {
            let shard_request = ShardQueryRequest {
                query_id: query.query_id.clone(),
                shard_id,
                query_vector: query.original_query.clone(),
                query_type: query.query_type.clone(),
                parameters: query.parameters.clone(),
                timeout_ms: 30000, // 30 second timeout
            };
            
            // Find the primary node for this shard
            let topology = self.topology.read().await;
            if let Some(shard) = topology.shards.get(&shard_id) {
                let node_client = self.get_or_create_client(&shard.primary_node).await?;
                
                let task = tokio::spawn(async move {
                    node_client.execute_shard_query(shard_request).await
                });
                
                query_tasks.push(task);
            }
        }
        
        // Wait for all shard queries to complete
        let results = join_all(query_tasks).await;
        
        // Collect successful results
        let mut shard_responses = Vec::new();
        for result in results {
            match result {
                Ok(Ok(response)) => shard_responses.push(response),
                Ok(Err(e)) => {
                    warn!("Shard query failed: {}", e);
                    // Could implement retry logic here
                }
                Err(e) => {
                    error!("Task failed: {}", e);
                }
            }
        }
        
        Ok(shard_responses)
    }
    
    async fn get_or_create_client(&self, node_id: &NodeId) -> Result<Arc<NodeClient>> {
        let clients = self.node_clients.read().await;
        
        if let Some(client) = clients.get(node_id) {
            return Ok(client.clone());
        }
        
        drop(clients);
        
        // Create new client
        let topology = self.topology.read().await;
        let node_info = topology.nodes.get(node_id)
            .ok_or_else(|| anyhow!("Node {} not found in topology", node_id))?;
        
        let client = Arc::new(NodeClient::new(node_id.clone(), node_info.address).await?);
        
        // Store client for reuse
        let mut clients = self.node_clients.write().await;
        clients.insert(node_id.clone(), client.clone());
        
        Ok(client)
    }
}

impl NodeClient {
    pub async fn new(node_id: NodeId, address: std::net::SocketAddr) -> Result<Self> {
        // In a real implementation, this would establish a gRPC connection
        Ok(Self {
            node_id,
            address,
            client: None, // Would be actual gRPC client
        })
    }
    
    pub async fn execute_shard_query(&self, request: ShardQueryRequest) -> Result<ShardQueryResponse> {
        let start_time = std::time::Instant::now();
        
        // Simulate query execution
        // In real implementation, this would make gRPC call to the node
        
        // Simulate processing time based on query complexity
        let processing_time = match request.query_type {
            QueryType::SimilaritySearch => {
                // Simulate vector similarity computation
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                
                // Generate mock results
                let k = request.parameters.k.unwrap_or(10);
                let results: Vec<(usize, f32)> = (0..k)
                    .map(|i| (i, 1.0 - (i as f32 * 0.1)))
                    .collect();
                
                ShardQueryResponse {
                    query_id: request.query_id,
                    shard_id: request.shard_id,
                    results,
                    status: "success".to_string(),
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error: None,
                }
            }
            QueryType::VectorInsertion => {
                tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
                
                ShardQueryResponse {
                    query_id: request.query_id,
                    shard_id: request.shard_id,
                    results: vec![(1, 1.0)], // Vector ID and success indicator
                    status: "inserted".to_string(),
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error: None,
                }
            }
            _ => {
                ShardQueryResponse {
                    query_id: request.query_id,
                    shard_id: request.shard_id,
                    results: vec![],
                    status: "completed".to_string(),
                    execution_time_ms: start_time.elapsed().as_millis() as u64,
                    error: None,
                }
            }
        };
        
        Ok(processing_time)
    }
}

impl ResultAggregator {
    pub fn new() -> Self {
        Self {}
    }
    
    /// Aggregate results from distributed query execution
    #[instrument(skip(self, query, step_results))]
    pub async fn aggregate_results(
        &self,
        query: &DistributedQuery,
        step_results: &HashMap<String, crate::coordinator::StepResult>,
    ) -> Result<QueryResultData> {
        match query.query_type {
            QueryType::SimilaritySearch => {
                let shard_results: Vec<serde_json::Value> = step_results
                    .values()
                    .filter_map(|result| result.result_data.clone())
                    .collect();
                
                let k = query.parameters.k.unwrap_or(10);
                let aggregated = self.aggregate_similarity_results(&shard_results, k).await?;
                
                // Convert to expected format
                if let serde_json::Value::Array(results) = aggregated {
                    let similarity_results: Vec<(usize, f32)> = results
                        .into_iter()
                        .filter_map(|v| {
                            if let serde_json::Value::Array(pair) = v {
                                if pair.len() == 2 {
                                    let idx = pair[0].as_u64()? as usize;
                                    let score = pair[1].as_f64()? as f32;
                                    Some((idx, score))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect();
                    
                    Ok(QueryResultData::SimilarityResults(similarity_results))
                } else {
                    Ok(QueryResultData::SimilarityResults(vec![]))
                }
            }
            QueryType::VectorInsertion => {
                Ok(QueryResultData::InsertionResult {
                    vector_id: 12345, // Would be actual vector ID
                    shard_id: query.target_shards.first().copied().unwrap_or(0),
                })
            }
            QueryType::ClusterStatus => {
                // Aggregate cluster information
                Ok(QueryResultData::ClusterInfo(ClusterInfo {
                    cluster_name: "spiraldelta-cluster".to_string(),
                    total_nodes: 3,
                    healthy_nodes: 3,
                    total_shards: 10,
                    active_shards: 10,
                    total_vectors: 1000000,
                    total_storage_bytes: 1024 * 1024 * 1024, // 1GB
                    resource_utilization: ClusterResources {
                        total_memory_gb: 32.0,
                        used_memory_gb: 16.0,
                        total_cpu_cores: 24,
                        cpu_utilization: 0.6,
                        total_storage_gb: 1000.0,
                        used_storage_gb: 100.0,
                    },
                    performance_metrics: ClusterPerformance {
                        queries_per_second: 1000.0,
                        avg_query_latency_ms: 50.0,
                        p95_latency_ms: 100.0,
                        p99_latency_ms: 200.0,
                        error_rate: 0.01,
                        throughput_vectors_per_second: 10000.0,
                    },
                }))
            }
            _ => Ok(QueryResultData::SimilarityResults(vec![])),
        }
    }
    
    pub async fn aggregate_similarity_results(
        &self,
        shard_results: &[serde_json::Value],
        k: usize,
    ) -> Result<serde_json::Value> {
        let mut all_results = Vec::new();
        
        // Extract results from each shard
        for shard_result in shard_results {
            if let Some(results) = shard_result.as_array() {
                for result in results {
                    if let Some(shard_response) = result.as_object() {
                        if let Some(serde_json::Value::Array(pairs)) = shard_response.get("results") {
                            for pair in pairs {
                                if let serde_json::Value::Array(vec) = pair {
                                    if vec.len() == 2 {
                                        if let (Some(idx), Some(score)) = (vec[0].as_u64(), vec[1].as_f64()) {
                                            all_results.push((idx as usize, score as f32));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Sort by score (descending) and take top-k
        all_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);
        
        // Convert back to JSON
        let result_json: Vec<serde_json::Value> = all_results
            .into_iter()
            .map(|(idx, score)| serde_json::json!([idx, score]))
            .collect();
        
        Ok(serde_json::Value::Array(result_json))
    }
    
    pub async fn aggregate_insertion_results(
        &self,
        _shard_results: &[serde_json::Value],
    ) -> Result<serde_json::Value> {
        // Placeholder for insertion result aggregation
        Ok(serde_json::json!({"status": "success"}))
    }
}