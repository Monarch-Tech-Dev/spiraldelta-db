/*!
Distributed Query Coordinator

Manages query planning, execution, and result aggregation across cluster nodes.
*/

use crate::*;
use anyhow::{Result, anyhow};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::time::{timeout, Duration};
use tracing::{info, warn, error, debug, instrument};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// Query coordinator responsible for distributed query execution
pub struct QueryCoordinator {
    /// Node information for this coordinator
    node_info: NodeInfo,
    
    /// Cluster configuration
    config: ClusterConfig,
    
    /// Cluster topology reference
    topology: Arc<RwLock<ClusterTopology>>,
    
    /// Query execution engine
    execution_engine: Arc<DistributedExecutionEngine>,
    
    /// Query planner
    planner: Arc<QueryPlanner>,
    
    /// Result aggregator
    aggregator: Arc<ResultAggregator>,
    
    /// Monitoring system
    monitoring: Arc<MonitoringSystem>,
    
    /// Active queries
    active_queries: Arc<RwLock<HashMap<QueryId, ActiveQuery>>>,
    
    /// Query queue for scheduling
    query_queue: Arc<RwLock<VecDeque<QueuedQuery>>>,
}

#[derive(Debug)]
struct ActiveQuery {
    query: DistributedQuery,
    execution_context: ExecutionContext,
    result_sender: oneshot::Sender<QueryResult>,
    started_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug)]
struct QueuedQuery {
    query: DistributedQuery,
    result_sender: oneshot::Sender<QueryResult>,
    queued_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug)]
struct ExecutionContext {
    plan: ExecutionPlan,
    step_results: HashMap<String, StepResult>,
    current_step: Option<String>,
    nodes_involved: Vec<NodeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StepResult {
    step_id: String,
    status: StepStatus,
    result_data: Option<serde_json::Value>,
    error: Option<String>,
    duration_ms: u64,
    node_stats: HashMap<NodeId, NodeQueryStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

impl QueryCoordinator {
    /// Create a new query coordinator
    pub async fn new(
        node_info: NodeInfo,
        config: ClusterConfig,
        topology: Arc<RwLock<ClusterTopology>>,
        monitoring: Arc<MonitoringSystem>,
    ) -> Result<Self> {
        let execution_engine = Arc::new(
            DistributedExecutionEngine::new(topology.clone(), monitoring.clone()).await?
        );
        
        let planner = Arc::new(QueryPlanner::new(config.clone(), topology.clone()));
        let aggregator = Arc::new(ResultAggregator::new());
        
        let coordinator = Self {
            node_info,
            config,
            topology,
            execution_engine,
            planner,
            aggregator,
            monitoring,
            active_queries: Arc::new(RwLock::new(HashMap::new())),
            query_queue: Arc::new(RwLock::new(VecDeque::new())),
        };
        
        // Start background tasks
        coordinator.start_background_tasks().await?;
        
        Ok(coordinator)
    }
    
    /// Execute a distributed query
    #[instrument(skip(self, query))]
    pub async fn execute_query(&self, query: DistributedQuery) -> Result<QueryResult> {
        let query_id = query.query_id.clone();
        info!("Executing distributed query: {}", query_id);
        
        // Check if query can be executed immediately or needs queuing
        if self.should_queue_query(&query).await? {
            self.queue_query(query).await
        } else {
            self.execute_query_immediate(query).await
        }
    }
    
    async fn execute_query_immediate(&self, query: DistributedQuery) -> Result<QueryResult> {
        let query_id = query.query_id.clone();
        let start_time = chrono::Utc::now();
        
        // Create result channel
        let (result_sender, result_receiver) = oneshot::channel();
        
        // Plan query execution
        let execution_plan = self.planner.plan_query(&query).await?;
        debug!("Generated execution plan for query {}: {} steps", query_id, execution_plan.steps.len());
        
        // Create execution context
        let execution_context = ExecutionContext {
            plan: execution_plan,
            step_results: HashMap::new(),
            current_step: None,
            nodes_involved: Vec::new(),
        };
        
        // Add to active queries
        {
            let mut active_queries = self.active_queries.write().await;
            active_queries.insert(query_id.clone(), ActiveQuery {
                query: query.clone(),
                execution_context,
                result_sender,
                started_at: start_time,
            });
        }
        
        // Execute query asynchronously
        let coordinator = Arc::new(self.clone());
        tokio::spawn(async move {
            let result = coordinator.execute_query_steps(query_id.clone()).await;
            coordinator.complete_query(query_id, result).await;
        });
        
        // Wait for result with timeout
        let timeout_duration = query.deadline
            .map(|deadline| {
                let now = chrono::Utc::now();
                if deadline > now {
                    Duration::from_millis((deadline - now).num_milliseconds() as u64)
                } else {
                    Duration::from_millis(0)
                }
            })
            .unwrap_or(Duration::from_secs(30));
        
        match timeout(timeout_duration, result_receiver).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Err(anyhow!("Query execution channel closed")),
            Err(_) => {
                // Query timed out
                self.cancel_query(&query.query_id).await?;
                Err(anyhow!("Query timed out"))
            }
        }
    }
    
    async fn execute_query_steps(&self, query_id: QueryId) -> Result<QueryResult> {
        let start_time = std::time::Instant::now();
        
        // Get query and execution context
        let (query, mut execution_context) = {
            let active_queries = self.active_queries.read().await;
            let active_query = active_queries.get(&query_id)
                .ok_or_else(|| anyhow!("Query {} not found in active queries", query_id))?;
            (active_query.query.clone(), active_query.execution_context.clone())
        };
        
        // Execute steps in dependency order
        let mut completed_steps = std::collections::HashSet::new();
        let mut step_results = HashMap::new();
        
        for step in &execution_context.plan.steps {
            // Check if all dependencies are completed
            if !step.dependencies.iter().all(|dep| completed_steps.contains(dep)) {
                continue; // Skip this step for now
            }
            
            debug!("Executing step: {} for query {}", step.step_id, query_id);
            
            // Update current step
            execution_context.current_step = Some(step.step_id.clone());
            
            // Execute the step
            let step_result = self.execute_step(&query, step, &step_results).await?;
            
            // Store result
            step_results.insert(step.step_id.clone(), step_result.clone());
            completed_steps.insert(step.step_id.clone());
            
            // Check if step failed
            if matches!(step_result.status, StepStatus::Failed) {
                return Ok(QueryResult {
                    query_id: query_id.clone(),
                    status: QueryStatus::Failed,
                    results: QueryResultData::SimilarityResults(vec![]),
                    statistics: self.create_query_statistics(&query, &step_results, start_time),
                    error: step_result.error,
                });
            }
        }
        
        // Aggregate final results
        let final_results = self.aggregator.aggregate_results(&query, &step_results).await?;
        
        Ok(QueryResult {
            query_id,
            status: QueryStatus::Completed,
            results: final_results,
            statistics: self.create_query_statistics(&query, &step_results, start_time),
            error: None,
        })
    }
    
    async fn execute_step(
        &self,
        query: &DistributedQuery,
        step: &ExecutionStep,
        previous_results: &HashMap<String, StepResult>,
    ) -> Result<StepResult> {
        let step_start = std::time::Instant::now();
        
        match step.step_type {
            StepType::ShardQuery => {
                self.execute_shard_query(query, step, previous_results).await
            }
            StepType::ResultAggregation => {
                self.execute_result_aggregation(query, step, previous_results).await
            }
            StepType::DataMigration => {
                self.execute_data_migration(query, step, previous_results).await
            }
            StepType::IndexUpdate => {
                self.execute_index_update(query, step, previous_results).await
            }
            StepType::Cleanup => {
                self.execute_cleanup(query, step, previous_results).await
            }
        }
    }
    
    async fn execute_shard_query(
        &self,
        query: &DistributedQuery,
        step: &ExecutionStep,
        _previous_results: &HashMap<String, StepResult>,
    ) -> Result<StepResult> {
        let step_start = std::time::Instant::now();
        
        // Execute query on target shards
        let shard_results = self.execution_engine.execute_on_shards(
            &query.target_shards,
            &step.target_nodes,
            query.clone(),
        ).await?;
        
        let duration = step_start.elapsed().as_millis() as u64;
        
        Ok(StepResult {
            step_id: step.step_id.clone(),
            status: StepStatus::Completed,
            result_data: Some(serde_json::to_value(&shard_results)?),
            error: None,
            duration_ms: duration,
            node_stats: HashMap::new(), // Would be populated by execution engine
        })
    }
    
    async fn execute_result_aggregation(
        &self,
        query: &DistributedQuery,
        step: &ExecutionStep,
        previous_results: &HashMap<String, StepResult>,
    ) -> Result<StepResult> {
        let step_start = std::time::Instant::now();
        
        // Collect results from previous steps
        let shard_results: Vec<serde_json::Value> = previous_results
            .values()
            .filter_map(|result| result.result_data.clone())
            .collect();
        
        // Aggregate results based on query type
        let aggregated_result = match query.query_type {
            QueryType::SimilaritySearch => {
                self.aggregator.aggregate_similarity_results(&shard_results, query.parameters.k.unwrap_or(10)).await?
            }
            QueryType::VectorInsertion => {
                self.aggregator.aggregate_insertion_results(&shard_results).await?
            }
            _ => serde_json::Value::Null,
        };
        
        let duration = step_start.elapsed().as_millis() as u64;
        
        Ok(StepResult {
            step_id: step.step_id.clone(),
            status: StepStatus::Completed,
            result_data: Some(aggregated_result),
            error: None,
            duration_ms: duration,
            node_stats: HashMap::new(),
        })
    }
    
    async fn execute_data_migration(
        &self,
        _query: &DistributedQuery,
        step: &ExecutionStep,
        _previous_results: &HashMap<String, StepResult>,
    ) -> Result<StepResult> {
        // Placeholder for data migration logic
        Ok(StepResult {
            step_id: step.step_id.clone(),
            status: StepStatus::Completed,
            result_data: None,
            error: None,
            duration_ms: 100,
            node_stats: HashMap::new(),
        })
    }
    
    async fn execute_index_update(
        &self,
        _query: &DistributedQuery,
        step: &ExecutionStep,
        _previous_results: &HashMap<String, StepResult>,
    ) -> Result<StepResult> {
        // Placeholder for index update logic
        Ok(StepResult {
            step_id: step.step_id.clone(),
            status: StepStatus::Completed,
            result_data: None,
            error: None,
            duration_ms: 50,
            node_stats: HashMap::new(),
        })
    }
    
    async fn execute_cleanup(
        &self,
        _query: &DistributedQuery,
        step: &ExecutionStep,
        _previous_results: &HashMap<String, StepResult>,
    ) -> Result<StepResult> {
        // Placeholder for cleanup logic
        Ok(StepResult {
            step_id: step.step_id.clone(),
            status: StepStatus::Completed,
            result_data: None,
            error: None,
            duration_ms: 10,
            node_stats: HashMap::new(),
        })
    }
    
    async fn should_queue_query(&self, query: &DistributedQuery) -> Result<bool> {
        // Check resource availability and current load
        let topology = self.topology.read().await;
        let active_queries = self.active_queries.read().await;
        
        // Check if we're at capacity
        if active_queries.len() >= 100 { // Max concurrent queries
            return Ok(true);
        }
        
        // Check if required nodes are available
        for shard_id in &query.target_shards {
            if let Some(shard) = topology.shards.get(shard_id) {
                let primary_node = topology.nodes.get(&shard.primary_node);
                if primary_node.map_or(true, |node| !matches!(node.status, NodeStatus::Healthy)) {
                    return Ok(true); // Queue until node is healthy
                }
            }
        }
        
        Ok(false)
    }
    
    async fn queue_query(&self, query: DistributedQuery) -> Result<QueryResult> {
        let (result_sender, result_receiver) = oneshot::channel();
        
        {
            let mut queue = self.query_queue.write().await;
            queue.push_back(QueuedQuery {
                query,
                result_sender,
                queued_at: chrono::Utc::now(),
            });
        }
        
        // Wait for result
        result_receiver.await
            .map_err(|_| anyhow!("Query queue channel closed"))
    }
    
    async fn cancel_query(&self, query_id: &QueryId) -> Result<()> {
        let mut active_queries = self.active_queries.write().await;
        if let Some(active_query) = active_queries.remove(query_id) {
            // Send cancellation result
            let _ = active_query.result_sender.send(QueryResult {
                query_id: query_id.clone(),
                status: QueryStatus::Cancelled,
                results: QueryResultData::SimilarityResults(vec![]),
                statistics: QueryStatistics {
                    total_duration_ms: 0,
                    nodes_involved: 0,
                    shards_queried: 0,
                    network_round_trips: 0,
                    data_transferred_bytes: 0,
                    per_node_stats: HashMap::new(),
                },
                error: Some("Query cancelled due to timeout".to_string()),
            });
        }
        Ok(())
    }
    
    async fn complete_query(&self, query_id: QueryId, result: Result<QueryResult>) {
        let mut active_queries = self.active_queries.write().await;
        if let Some(active_query) = active_queries.remove(&query_id) {
            let final_result = result.unwrap_or_else(|e| QueryResult {
                query_id: query_id.clone(),
                status: QueryStatus::Failed,
                results: QueryResultData::SimilarityResults(vec![]),
                statistics: QueryStatistics {
                    total_duration_ms: 0,
                    nodes_involved: 0,
                    shards_queried: 0,
                    network_round_trips: 0,
                    data_transferred_bytes: 0,
                    per_node_stats: HashMap::new(),
                },
                error: Some(e.to_string()),
            });
            
            let _ = active_query.result_sender.send(final_result);
        }
    }
    
    fn create_query_statistics(
        &self,
        query: &DistributedQuery,
        step_results: &HashMap<String, StepResult>,
        start_time: std::time::Instant,
    ) -> QueryStatistics {
        let total_duration = start_time.elapsed().as_millis() as u64;
        let nodes_involved = step_results.values()
            .flat_map(|step| step.node_stats.keys())
            .collect::<std::collections::HashSet<_>>()
            .len();
        
        let shards_queried = query.target_shards.len();
        
        // Aggregate node statistics
        let mut per_node_stats = HashMap::new();
        for step_result in step_results.values() {
            for (node_id, node_stat) in &step_result.node_stats {
                let entry = per_node_stats.entry(node_id.clone()).or_insert(NodeQueryStats {
                    duration_ms: 0,
                    memory_used_mb: 0.0,
                    cpu_time_ms: 0,
                    network_io_bytes: 0,
                });
                
                entry.duration_ms += node_stat.duration_ms;
                entry.memory_used_mb += node_stat.memory_used_mb;
                entry.cpu_time_ms += node_stat.cpu_time_ms;
                entry.network_io_bytes += node_stat.network_io_bytes;
            }
        }
        
        QueryStatistics {
            total_duration_ms: total_duration,
            nodes_involved,
            shards_queried,
            network_round_trips: step_results.len(),
            data_transferred_bytes: per_node_stats.values()
                .map(|stats| stats.network_io_bytes)
                .sum(),
            per_node_stats,
        }
    }
    
    async fn start_background_tasks(&self) -> Result<()> {
        // Start query queue processor
        let coordinator = Arc::new(self.clone());
        tokio::spawn(async move {
            coordinator.process_query_queue().await;
        });
        
        // Start health checker
        let coordinator = Arc::new(self.clone());
        tokio::spawn(async move {
            coordinator.check_cluster_health().await;
        });
        
        Ok(())
    }
    
    async fn process_query_queue(&self) {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        
        loop {
            interval.tick().await;
            
            // Try to dequeue and execute queries
            let queued_query = {
                let mut queue = self.query_queue.write().await;
                queue.pop_front()
            };
            
            if let Some(queued_query) = queued_query {
                // Check if query can be executed now
                if !self.should_queue_query(&queued_query.query).await.unwrap_or(true) {
                    // Execute query
                    let result = self.execute_query_immediate(queued_query.query).await;
                    let _ = queued_query.result_sender.send(result.unwrap_or_else(|e| QueryResult {
                        query_id: Uuid::new_v4().to_string(),
                        status: QueryStatus::Failed,
                        results: QueryResultData::SimilarityResults(vec![]),
                        statistics: QueryStatistics {
                            total_duration_ms: 0,
                            nodes_involved: 0,
                            shards_queried: 0,
                            network_round_trips: 0,
                            data_transferred_bytes: 0,
                            per_node_stats: HashMap::new(),
                        },
                        error: Some(e.to_string()),
                    }));
                } else {
                    // Put back in queue
                    let mut queue = self.query_queue.write().await;
                    queue.push_front(queued_query);
                }
            }
        }
    }
    
    async fn check_cluster_health(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            // Check node health and update topology
            // This would involve pinging nodes and updating their status
            
            // Check for failed queries and retry if needed
            let failed_queries: Vec<QueryId> = {
                let active_queries = self.active_queries.read().await;
                active_queries.iter()
                    .filter(|(_, query)| {
                        // Check if query has been running too long
                        let elapsed = chrono::Utc::now() - query.started_at;
                        elapsed.num_seconds() > 300 // 5 minutes
                    })
                    .map(|(id, _)| id.clone())
                    .collect()
            };
            
            for query_id in failed_queries {
                warn!("Query {} has been running too long, cancelling", query_id);
                let _ = self.cancel_query(&query_id).await;
            }
        }
    }
}

// Clone implementation for Arc usage in async tasks
impl Clone for QueryCoordinator {
    fn clone(&self) -> Self {
        Self {
            node_info: self.node_info.clone(),
            config: self.config.clone(),
            topology: self.topology.clone(),
            execution_engine: self.execution_engine.clone(),
            planner: self.planner.clone(),
            aggregator: self.aggregator.clone(),
            monitoring: self.monitoring.clone(),
            active_queries: self.active_queries.clone(),
            query_queue: self.query_queue.clone(),
        }
    }
}