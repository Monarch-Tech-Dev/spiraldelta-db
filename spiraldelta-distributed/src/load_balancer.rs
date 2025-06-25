/*!
Load Balancer for Distributed System

Implements various load balancing strategies for distributing requests across cluster nodes.
*/

use crate::*;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug, instrument};
use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};

/// Load balancer for distributing requests across cluster nodes
pub struct LoadBalancer {
    /// Load balancing configuration
    config: LoadBalancerConfig,
    
    /// Current node weights and states
    node_states: Arc<RwLock<HashMap<NodeId, NodeState>>>,
    
    /// Round-robin counter for round-robin strategy
    round_robin_counter: Arc<RwLock<usize>>,
    
    /// Consistent hash ring for consistent hashing
    hash_ring: Arc<RwLock<consistent_hash::ConsistentHash<NodeId>>>,
    
    /// Circuit breakers for each node
    circuit_breakers: Arc<RwLock<HashMap<NodeId, CircuitBreaker>>>,
    
    /// Health checker
    health_checker: Arc<HealthChecker>,
}

#[derive(Debug, Clone)]
struct NodeState {
    node_info: NodeInfo,
    current_load: f64,
    avg_response_time_ms: f64,
    success_rate: f64,
    active_connections: u32,
    last_health_check: chrono::DateTime<chrono::Utc>,
    weight: f64,
}

#[derive(Debug, Clone)]
struct CircuitBreaker {
    state: CircuitBreakerState,
    failure_count: u32,
    success_count: u32,
    last_failure: Option<Instant>,
    next_attempt: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq)]
enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Health checker for monitoring node health
pub struct HealthChecker {
    config: LoadBalancerConfig,
    node_states: Arc<RwLock<HashMap<NodeId, NodeState>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time_ms: f64,
    pub node_distributions: HashMap<NodeId, u64>,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub async fn new(config: LoadBalancerConfig) -> Result<Self> {
        let node_states = Arc::new(RwLock::new(HashMap::new()));
        let hash_ring = Arc::new(RwLock::new(consistent_hash::ConsistentHash::new()));
        let circuit_breakers = Arc::new(RwLock::new(HashMap::new()));
        
        let health_checker = Arc::new(HealthChecker::new(config.clone(), node_states.clone()));
        
        let load_balancer = Self {
            config,
            node_states,
            round_robin_counter: Arc::new(RwLock::new(0)),
            hash_ring,
            circuit_breakers,
            health_checker,
        };
        
        // Start health checking
        load_balancer.start_health_checking().await?;
        
        Ok(load_balancer)
    }
    
    /// Select a node for request routing based on the configured strategy
    #[instrument(skip(self, available_nodes))]
    pub async fn select_node(&self, available_nodes: &[&NodeInfo]) -> Result<&NodeInfo> {
        if available_nodes.is_empty() {
            return Err(anyhow!("No available nodes"));
        }
        
        // Filter out nodes with open circuit breakers
        let healthy_nodes = self.filter_healthy_nodes(available_nodes).await?;
        
        if healthy_nodes.is_empty() {
            return Err(anyhow!("No healthy nodes available"));
        }
        
        match self.config.strategy {
            LoadBalancingStrategy::RoundRobin => {
                self.round_robin_select(&healthy_nodes).await
            }
            LoadBalancingStrategy::LeastConnections => {
                self.least_connections_select(&healthy_nodes).await
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.weighted_round_robin_select(&healthy_nodes).await
            }
            LoadBalancingStrategy::ConsistentHashing => {
                self.consistent_hash_select(&healthy_nodes, "default_key").await
            }
            LoadBalancingStrategy::LoadAware => {
                self.load_aware_select(&healthy_nodes).await
            }
        }
    }
    
    /// Select node for a specific key (useful for consistent hashing)
    #[instrument(skip(self, available_nodes))]
    pub async fn select_node_for_key(&self, available_nodes: &[&NodeInfo], key: &str) -> Result<&NodeInfo> {
        let healthy_nodes = self.filter_healthy_nodes(available_nodes).await?;
        
        if healthy_nodes.is_empty() {
            return Err(anyhow!("No healthy nodes available"));
        }
        
        match self.config.strategy {
            LoadBalancingStrategy::ConsistentHashing => {
                self.consistent_hash_select(&healthy_nodes, key).await
            }
            _ => self.select_node(available_nodes).await,
        }
    }
    
    async fn filter_healthy_nodes<'a>(&self, nodes: &[&'a NodeInfo]) -> Result<Vec<&'a NodeInfo>> {
        let circuit_breakers = self.circuit_breakers.read().await;
        let node_states = self.node_states.read().await;
        
        let healthy_nodes: Vec<&NodeInfo> = nodes.iter()
            .filter(|node| {
                // Check circuit breaker state
                if let Some(breaker) = circuit_breakers.get(&node.node_id) {
                    if breaker.state == CircuitBreakerState::Open {
                        return false;
                    }
                }
                
                // Check node health
                if !matches!(node.status, NodeStatus::Healthy) {
                    return false;
                }
                
                // Check last health check
                if let Some(state) = node_states.get(&node.node_id) {
                    let elapsed = chrono::Utc::now() - state.last_health_check;
                    if elapsed.num_seconds() > 120 { // 2 minutes
                        return false;
                    }
                }
                
                true
            })
            .copied()
            .collect();
        
        Ok(healthy_nodes)
    }
    
    async fn round_robin_select<'a>(&self, nodes: &[&'a NodeInfo]) -> Result<&'a NodeInfo> {
        let mut counter = self.round_robin_counter.write().await;
        let selected_index = *counter % nodes.len();
        *counter = (*counter + 1) % nodes.len();
        
        Ok(nodes[selected_index])
    }
    
    async fn least_connections_select<'a>(&self, nodes: &[&'a NodeInfo]) -> Result<&'a NodeInfo> {
        let node_states = self.node_states.read().await;
        
        let mut min_connections = u32::MAX;
        let mut selected_node = nodes[0];
        
        for &node in nodes {
            let connections = node_states.get(&node.node_id)
                .map(|state| state.active_connections)
                .unwrap_or(0);
            
            if connections < min_connections {
                min_connections = connections;
                selected_node = node;
            }
        }
        
        Ok(selected_node)
    }
    
    async fn weighted_round_robin_select<'a>(&self, nodes: &[&'a NodeInfo]) -> Result<&'a NodeInfo> {
        let node_states = self.node_states.read().await;
        
        // Calculate weighted selection
        let mut total_weight = 0.0;
        let weights: Vec<f64> = nodes.iter().map(|node| {
            let weight = node_states.get(&node.node_id)
                .map(|state| state.weight)
                .unwrap_or(1.0);
            total_weight += weight;
            weight
        }).collect();
        
        // Generate random number for selection
        let mut rng = rand::thread_rng();
        let random_weight = rand::Rng::gen_range(&mut rng, 0.0..total_weight);
        
        let mut current_weight = 0.0;
        for (i, &weight) in weights.iter().enumerate() {
            current_weight += weight;
            if random_weight <= current_weight {
                return Ok(nodes[i]);
            }
        }
        
        // Fallback to first node
        Ok(nodes[0])
    }
    
    async fn consistent_hash_select<'a>(&self, nodes: &[&'a NodeInfo], key: &str) -> Result<&'a NodeInfo> {
        let hash_ring = self.hash_ring.read().await;
        
        // If hash ring is empty, populate it with current nodes
        if hash_ring.is_empty() {
            drop(hash_ring);
            let mut hash_ring = self.hash_ring.write().await;
            for &node in nodes {
                hash_ring.add(&node.node_id, 1);
            }
        }
        
        let hash_ring = self.hash_ring.read().await;
        let selected_node_id = hash_ring.get(key)
            .ok_or_else(|| anyhow!("No node found in hash ring"))?;
        
        // Find the actual node info
        nodes.iter()
            .find(|node| &node.node_id == selected_node_id)
            .copied()
            .ok_or_else(|| anyhow!("Selected node not found in available nodes"))
    }
    
    async fn load_aware_select<'a>(&self, nodes: &[&'a NodeInfo]) -> Result<&'a NodeInfo> {
        let node_states = self.node_states.read().await;
        
        let mut best_score = f64::MAX;
        let mut selected_node = nodes[0];
        
        for &node in nodes {
            let state = node_states.get(&node.node_id);
            
            // Calculate load score (lower is better)
            let load_score = if let Some(state) = state {
                // Weighted combination of factors
                let load_factor = state.current_load * 0.4;
                let response_time_factor = (state.avg_response_time_ms / 1000.0) * 0.3;
                let connection_factor = (state.active_connections as f64 / 100.0) * 0.2;
                let reliability_factor = (1.0 - state.success_rate) * 0.1;
                
                load_factor + response_time_factor + connection_factor + reliability_factor
            } else {
                // No state available, give neutral score
                0.5
            };
            
            if load_score < best_score {
                best_score = load_score;
                selected_node = node;
            }
        }
        
        Ok(selected_node)
    }
    
    /// Update node state after a request
    #[instrument(skip(self))]
    pub async fn update_node_state(
        &self,
        node_id: &NodeId,
        response_time_ms: u64,
        success: bool,
    ) -> Result<()> {
        let mut node_states = self.node_states.write().await;
        let mut circuit_breakers = self.circuit_breakers.write().await;
        
        // Update node state
        if let Some(state) = node_states.get_mut(node_id) {
            // Update running averages
            let alpha = 0.1; // Smoothing factor
            state.avg_response_time_ms = 
                alpha * response_time_ms as f64 + (1.0 - alpha) * state.avg_response_time_ms;
            
            // Update success rate
            state.success_rate = alpha * (if success { 1.0 } else { 0.0 }) + (1.0 - alpha) * state.success_rate;
            
            // Adjust weight based on performance
            state.weight = self.calculate_node_weight(state).await;
        }
        
        // Update circuit breaker
        let breaker = circuit_breakers.entry(node_id.clone())
            .or_insert_with(|| CircuitBreaker {
                state: CircuitBreakerState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure: None,
                next_attempt: None,
            });
        
        if success {
            breaker.success_count += 1;
            breaker.failure_count = 0; // Reset failure count on success
            
            // Close circuit if enough successes in half-open state
            if breaker.state == CircuitBreakerState::HalfOpen && 
               breaker.success_count >= self.config.success_threshold {
                breaker.state = CircuitBreakerState::Closed;
                breaker.success_count = 0;
                info!("Circuit breaker for node {} closed", node_id);
            }
        } else {
            breaker.failure_count += 1;
            breaker.last_failure = Some(Instant::now());
            
            // Open circuit if failure threshold exceeded
            if breaker.state == CircuitBreakerState::Closed && 
               breaker.failure_count >= self.config.failure_threshold {
                breaker.state = CircuitBreakerState::Open;
                breaker.next_attempt = Some(Instant::now() + Duration::from_secs(self.config.timeout_seconds));
                warn!("Circuit breaker for node {} opened", node_id);
            }
        }
        
        Ok(())
    }
    
    async fn calculate_node_weight(&self, state: &NodeState) -> f64 {
        // Calculate weight based on node performance
        let base_weight = 1.0;
        let performance_factor = state.success_rate * 2.0; // 0.0 to 2.0
        let load_factor = 2.0 - state.current_load.min(2.0); // Higher load = lower weight
        let response_factor = 2.0 - (state.avg_response_time_ms / 500.0).min(2.0); // Normalize around 500ms
        
        (base_weight * performance_factor * load_factor * response_factor).max(0.1).min(3.0)
    }
    
    /// Add a node to the load balancer
    pub async fn add_node(&self, node_info: NodeInfo) -> Result<()> {
        let mut node_states = self.node_states.write().await;
        let mut hash_ring = self.hash_ring.write().await;
        
        // Add to node states
        node_states.insert(node_info.node_id.clone(), NodeState {
            node_info: node_info.clone(),
            current_load: 0.0,
            avg_response_time_ms: 100.0, // Default assumption
            success_rate: 1.0, // Optimistic start
            active_connections: 0,
            last_health_check: chrono::Utc::now(),
            weight: 1.0,
        });
        
        // Add to hash ring for consistent hashing
        hash_ring.add(&node_info.node_id, 1);
        
        info!("Added node {} to load balancer", node_info.node_id);
        Ok(())
    }
    
    /// Remove a node from the load balancer
    pub async fn remove_node(&self, node_id: &NodeId) -> Result<()> {
        let mut node_states = self.node_states.write().await;
        let mut hash_ring = self.hash_ring.write().await;
        let mut circuit_breakers = self.circuit_breakers.write().await;
        
        // Remove from all collections
        node_states.remove(node_id);
        hash_ring.remove(node_id);
        circuit_breakers.remove(node_id);
        
        info!("Removed node {} from load balancer", node_id);
        Ok(())
    }
    
    async fn start_health_checking(&self) -> Result<()> {
        let health_checker = self.health_checker.clone();
        let circuit_breakers = self.circuit_breakers.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                Duration::from_secs(config.health_check_interval_s)
            );
            
            loop {
                interval.tick().await;
                
                if let Err(e) = health_checker.check_all_nodes().await {
                    warn!("Health check failed: {}", e);
                }
                
                // Check for circuit breaker timeouts
                Self::check_circuit_breaker_timeouts(&circuit_breakers, &config).await;
            }
        });
        
        Ok(())
    }
    
    async fn check_circuit_breaker_timeouts(
        circuit_breakers: &Arc<RwLock<HashMap<NodeId, CircuitBreaker>>>,
        config: &LoadBalancerConfig,
    ) {
        let mut breakers = circuit_breakers.write().await;
        let now = Instant::now();
        
        for (node_id, breaker) in breakers.iter_mut() {
            if breaker.state == CircuitBreakerState::Open {
                if let Some(next_attempt) = breaker.next_attempt {
                    if now >= next_attempt {
                        breaker.state = CircuitBreakerState::HalfOpen;
                        breaker.success_count = 0;
                        info!("Circuit breaker for node {} moved to half-open", node_id);
                    }
                }
            }
        }
    }
    
    /// Get load balancing metrics
    pub async fn get_metrics(&self) -> Result<LoadBalancingMetrics> {
        let node_states = self.node_states.read().await;
        
        let total_requests: u64 = node_states.values()
            .map(|state| (state.success_rate * 1000.0) as u64) // Rough estimate
            .sum();
        
        let node_distributions: HashMap<NodeId, u64> = node_states.iter()
            .map(|(id, state)| (id.clone(), (state.success_rate * 100.0) as u64))
            .collect();
        
        let avg_response_time = if node_states.len() > 0 {
            node_states.values()
                .map(|state| state.avg_response_time_ms)
                .sum::<f64>() / node_states.len() as f64
        } else {
            0.0
        };
        
        Ok(LoadBalancingMetrics {
            total_requests,
            successful_requests: (total_requests as f64 * 0.95) as u64, // Estimate
            failed_requests: (total_requests as f64 * 0.05) as u64, // Estimate
            avg_response_time_ms: avg_response_time,
            node_distributions,
        })
    }
}

impl HealthChecker {
    fn new(config: LoadBalancerConfig, node_states: Arc<RwLock<HashMap<NodeId, NodeState>>>) -> Self {
        Self {
            config,
            node_states,
        }
    }
    
    async fn check_all_nodes(&self) -> Result<()> {
        let node_states = self.node_states.read().await;
        let node_ids: Vec<NodeId> = node_states.keys().cloned().collect();
        drop(node_states);
        
        for node_id in node_ids {
            if let Err(e) = self.check_node_health(&node_id).await {
                warn!("Health check failed for node {}: {}", node_id, e);
            }
        }
        
        Ok(())
    }
    
    async fn check_node_health(&self, node_id: &NodeId) -> Result<()> {
        // In a real implementation, this would make HTTP/gRPC health check calls
        // For now, we'll simulate health checks
        
        let mut node_states = self.node_states.write().await;
        if let Some(state) = node_states.get_mut(node_id) {
            // Simulate health check
            let health_ok = rand::random::<f64>() > 0.05; // 95% success rate
            
            if health_ok {
                state.last_health_check = chrono::Utc::now();
                state.current_load = rand::random::<f64>() * 0.8; // Random load 0-80%
            } else {
                // Mark as unhealthy if health check fails
                warn!("Health check failed for node {}", node_id);
            }
        }
        
        Ok(())
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

// We need to add the rand crate for the weighted selection
use rand::Rng;