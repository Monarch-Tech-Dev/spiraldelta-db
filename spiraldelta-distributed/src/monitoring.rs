/*!
Distributed Monitoring System

Provides comprehensive monitoring, metrics collection, and observability for the distributed cluster.
*/

use crate::*;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug, instrument};
use serde::{Serialize, Deserialize};
use prometheus::{Registry, Counter, Histogram, Gauge, IntGauge};
use std::time::{Duration, Instant};

/// Comprehensive monitoring system for distributed cluster
pub struct MonitoringSystem {
    /// Monitoring configuration
    config: MonitoringConfig,
    
    /// Prometheus metrics registry
    metrics_registry: Arc<Registry>,
    
    /// Cluster metrics
    cluster_metrics: Arc<ClusterMetrics>,
    
    /// Node metrics storage
    node_metrics: Arc<RwLock<HashMap<NodeId, NodeMetrics>>>,
    
    /// Query metrics storage
    query_metrics: Arc<RwLock<QueryMetricsAggregator>>,
    
    /// Alert manager
    alert_manager: Arc<AlertManager>,
    
    /// Metrics exporter
    metrics_exporter: Option<Arc<MetricsExporter>>,
}

/// Prometheus metrics for the cluster
pub struct ClusterMetrics {
    // Request metrics
    pub requests_total: Counter,
    pub requests_duration: Histogram,
    pub active_requests: IntGauge,
    
    // Node metrics
    pub nodes_total: IntGauge,
    pub healthy_nodes: IntGauge,
    pub node_cpu_usage: Gauge,
    pub node_memory_usage: Gauge,
    
    // Query metrics
    pub queries_total: Counter,
    pub query_duration: Histogram,
    pub query_errors: Counter,
    
    // Storage metrics
    pub vectors_total: IntGauge,
    pub storage_bytes: IntGauge,
    pub shards_total: IntGauge,
    
    // Network metrics
    pub network_bytes_sent: Counter,
    pub network_bytes_received: Counter,
}

#[derive(Debug, Clone)]
struct NodeMetrics {
    node_id: NodeId,
    cpu_usage: f64,
    memory_usage_gb: f64,
    disk_usage_gb: f64,
    network_io_bytes: u64,
    active_connections: u32,
    queries_per_second: f64,
    avg_response_time_ms: f64,
    error_rate: f64,
    last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Default)]
struct QueryMetricsAggregator {
    total_queries: u64,
    successful_queries: u64,
    failed_queries: u64,
    total_duration_ms: u64,
    query_types: HashMap<QueryType, QueryTypeMetrics>,
}

#[derive(Debug, Default, Clone)]
struct QueryTypeMetrics {
    count: u64,
    total_duration_ms: u64,
    error_count: u64,
    min_duration_ms: u64,
    max_duration_ms: u64,
}

/// Alert manager for handling cluster alerts
pub struct AlertManager {
    config: AlertConfig,
    active_alerts: Arc<RwLock<HashMap<String, Alert>>>,
    alert_rules: Vec<AlertRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub enable_alerts: bool,
    pub webhook_url: Option<String>,
    pub email_notifications: bool,
    pub alert_cooldown_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Alert {
    id: String,
    level: AlertLevel,
    title: String,
    description: String,
    node_id: Option<NodeId>,
    created_at: chrono::DateTime<chrono::Utc>,
    resolved_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum AlertLevel {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone)]
struct AlertRule {
    id: String,
    name: String,
    condition: AlertCondition,
    threshold: f64,
    level: AlertLevel,
    cooldown: Duration,
    last_triggered: Option<Instant>,
}

#[derive(Debug, Clone)]
enum AlertCondition {
    HighCpuUsage,
    HighMemoryUsage,
    HighErrorRate,
    SlowResponseTime,
    NodeDown,
    LowDiskSpace,
}

/// Metrics exporter for external monitoring systems
pub struct MetricsExporter {
    config: MonitoringConfig,
    registry: Arc<Registry>,
}

impl MonitoringSystem {
    /// Create a new monitoring system
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        let metrics_registry = Arc::new(Registry::new());
        let cluster_metrics = Arc::new(ClusterMetrics::new(&metrics_registry)?);
        let alert_manager = Arc::new(AlertManager::new(AlertConfig::default()).await?);
        
        let metrics_exporter = if config.enable_metrics {
            Some(Arc::new(MetricsExporter::new(config.clone(), metrics_registry.clone()).await?))
        } else {
            None
        };
        
        let monitoring = Self {
            config,
            metrics_registry,
            cluster_metrics,
            node_metrics: Arc::new(RwLock::new(HashMap::new())),
            query_metrics: Arc::new(RwLock::new(QueryMetricsAggregator::default())),
            alert_manager,
            metrics_exporter,
        };
        
        // Start background monitoring tasks
        monitoring.start_background_tasks().await?;
        
        Ok(monitoring)
    }
    
    /// Record a request metric
    #[instrument(skip(self))]
    pub async fn record_request_metrics(&self, duration_ms: u64, status_code: u16) -> Result<()> {
        // Update Prometheus metrics
        self.cluster_metrics.requests_total.inc();
        self.cluster_metrics.requests_duration.observe(duration_ms as f64 / 1000.0);
        
        // Record error if status indicates failure
        if status_code >= 400 {
            self.cluster_metrics.query_errors.inc();
        }
        
        Ok(())
    }
    
    /// Record query execution metrics
    #[instrument(skip(self))]
    pub async fn record_query_metrics(
        &self,
        query_type: QueryType,
        duration_ms: u64,
        success: bool,
    ) -> Result<()> {
        // Update Prometheus metrics
        self.cluster_metrics.queries_total.inc();
        self.cluster_metrics.query_duration.observe(duration_ms as f64 / 1000.0);
        
        if !success {
            self.cluster_metrics.query_errors.inc();
        }
        
        // Update aggregated metrics
        let mut query_metrics = self.query_metrics.write().await;
        query_metrics.total_queries += 1;
        query_metrics.total_duration_ms += duration_ms;
        
        if success {
            query_metrics.successful_queries += 1;
        } else {
            query_metrics.failed_queries += 1;
        }
        
        // Update per-type metrics
        let type_metrics = query_metrics.query_types.entry(query_type).or_default();
        type_metrics.count += 1;
        type_metrics.total_duration_ms += duration_ms;
        
        if duration_ms < type_metrics.min_duration_ms || type_metrics.min_duration_ms == 0 {
            type_metrics.min_duration_ms = duration_ms;
        }
        if duration_ms > type_metrics.max_duration_ms {
            type_metrics.max_duration_ms = duration_ms;
        }
        
        if !success {
            type_metrics.error_count += 1;
        }
        
        Ok(())
    }
    
    /// Record node metrics
    #[instrument(skip(self))]
    pub async fn record_node_metrics(&self, node_id: &NodeId, metrics: NodeMetrics) -> Result<()> {
        // Update Prometheus gauges
        self.cluster_metrics.node_cpu_usage.set(metrics.cpu_usage);
        self.cluster_metrics.node_memory_usage.set(metrics.memory_usage_gb);
        
        // Store node metrics
        let mut node_metrics_store = self.node_metrics.write().await;
        node_metrics_store.insert(node_id.clone(), metrics.clone());
        
        // Check for alert conditions
        self.check_node_alerts(node_id, &metrics).await?;
        
        Ok(())
    }
    
    /// Report node health status
    #[instrument(skip(self))]
    pub async fn report_node_health(&self, node_id: &NodeId) -> Result<()> {
        // Collect current node metrics
        let node_metrics = self.collect_node_metrics(node_id).await?;
        self.record_node_metrics(node_id, node_metrics).await?;
        
        Ok(())
    }
    
    /// Record gateway metrics
    pub async fn record_gateway_metrics(&self, active_connections: usize) -> Result<()> {
        self.cluster_metrics.active_requests.set(active_connections as i64);
        Ok(())
    }
    
    /// Track topology changes
    pub async fn track_topology_change(&self, change_type: &str) -> Result<()> {
        info!("Topology change recorded: {}", change_type);
        
        // Update cluster metrics based on change type
        match change_type {
            "node_added" => {
                // This would be updated when we have the actual count
                info!("Node added to cluster");
            }
            "node_removed" => {
                info!("Node removed from cluster");
            }
            _ => {}
        }
        
        Ok(())
    }
    
    /// Get cluster performance metrics
    pub async fn get_cluster_performance(&self) -> Result<ClusterPerformance> {
        let query_metrics = self.query_metrics.read().await;
        
        let avg_latency = if query_metrics.total_queries > 0 {
            query_metrics.total_duration_ms as f64 / query_metrics.total_queries as f64
        } else {
            0.0
        };
        
        let error_rate = if query_metrics.total_queries > 0 {
            query_metrics.failed_queries as f64 / query_metrics.total_queries as f64
        } else {
            0.0
        };
        
        // Calculate QPS (queries per second) - simplified calculation
        let queries_per_second = 100.0; // Placeholder - would calculate from time windows
        
        Ok(ClusterPerformance {
            queries_per_second,
            avg_query_latency_ms: avg_latency,
            p95_latency_ms: avg_latency * 1.5, // Estimate
            p99_latency_ms: avg_latency * 2.0, // Estimate
            error_rate,
            throughput_vectors_per_second: queries_per_second * 10.0, // Estimate
        })
    }
    
    /// Get cluster statistics
    pub async fn get_cluster_statistics(&self) -> Result<HashMap<String, serde_json::Value>> {
        let query_metrics = self.query_metrics.read().await;
        let node_metrics = self.node_metrics.read().await;
        
        let mut stats = HashMap::new();
        
        // Query statistics
        stats.insert("total_queries".to_string(), query_metrics.total_queries.into());
        stats.insert("successful_queries".to_string(), query_metrics.successful_queries.into());
        stats.insert("failed_queries".to_string(), query_metrics.failed_queries.into());
        
        // Node statistics
        stats.insert("total_nodes".to_string(), node_metrics.len().into());
        
        let avg_cpu = if node_metrics.len() > 0 {
            node_metrics.values().map(|m| m.cpu_usage).sum::<f64>() / node_metrics.len() as f64
        } else {
            0.0
        };
        stats.insert("avg_cpu_usage".to_string(), avg_cpu.into());
        
        let total_memory = node_metrics.values().map(|m| m.memory_usage_gb).sum::<f64>();
        stats.insert("total_memory_gb".to_string(), total_memory.into());
        
        Ok(stats)
    }
    
    /// Get average response time
    pub async fn get_avg_response_time(&self) -> Result<f64> {
        let query_metrics = self.query_metrics.read().await;
        
        if query_metrics.total_queries > 0 {
            Ok(query_metrics.total_duration_ms as f64 / query_metrics.total_queries as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Get error rate
    pub async fn get_error_rate(&self) -> Result<f64> {
        let query_metrics = self.query_metrics.read().await;
        
        if query_metrics.total_queries > 0 {
            Ok(query_metrics.failed_queries as f64 / query_metrics.total_queries as f64)
        } else {
            Ok(0.0)
        }
    }
    
    async fn collect_node_metrics(&self, node_id: &NodeId) -> Result<NodeMetrics> {
        // In a real implementation, this would collect actual system metrics
        // For now, we'll simulate metrics collection
        
        Ok(NodeMetrics {
            node_id: node_id.clone(),
            cpu_usage: rand::random::<f64>() * 0.8, // 0-80%
            memory_usage_gb: 2.0 + rand::random::<f64>() * 6.0, // 2-8 GB
            disk_usage_gb: 10.0 + rand::random::<f64>() * 40.0, // 10-50 GB
            network_io_bytes: (rand::random::<u64>() % 1000000) + 1000, // Random network I/O
            active_connections: (rand::random::<u32>() % 100) + 10, // 10-110 connections
            queries_per_second: 50.0 + rand::random::<f64>() * 100.0, // 50-150 QPS
            avg_response_time_ms: 20.0 + rand::random::<f64>() * 80.0, // 20-100ms
            error_rate: rand::random::<f64>() * 0.05, // 0-5% error rate
            last_updated: chrono::Utc::now(),
        })
    }
    
    async fn check_node_alerts(&self, node_id: &NodeId, metrics: &NodeMetrics) -> Result<()> {
        // Check various alert conditions
        if metrics.cpu_usage > 0.9 {
            self.alert_manager.trigger_alert(
                format!("high_cpu_{}", node_id),
                AlertLevel::Warning,
                "High CPU Usage".to_string(),
                format!("Node {} CPU usage is {:.1}%", node_id, metrics.cpu_usage * 100.0),
                Some(node_id.clone()),
            ).await?;
        }
        
        if metrics.memory_usage_gb > 7.0 {
            self.alert_manager.trigger_alert(
                format!("high_memory_{}", node_id),
                AlertLevel::Warning,
                "High Memory Usage".to_string(),
                format!("Node {} memory usage is {:.1} GB", node_id, metrics.memory_usage_gb),
                Some(node_id.clone()),
            ).await?;
        }
        
        if metrics.error_rate > 0.1 {
            self.alert_manager.trigger_alert(
                format!("high_error_rate_{}", node_id),
                AlertLevel::Critical,
                "High Error Rate".to_string(),
                format!("Node {} error rate is {:.1}%", node_id, metrics.error_rate * 100.0),
                Some(node_id.clone()),
            ).await?;
        }
        
        Ok(())
    }
    
    async fn start_background_tasks(&self) -> Result<()> {
        // Start metrics collection task
        let node_metrics = self.node_metrics.clone();
        let cluster_metrics = self.cluster_metrics.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                
                let metrics = node_metrics.read().await;
                let healthy_nodes = metrics.values()
                    .filter(|m| m.cpu_usage < 0.9 && m.error_rate < 0.1)
                    .count();
                
                cluster_metrics.nodes_total.set(metrics.len() as i64);
                cluster_metrics.healthy_nodes.set(healthy_nodes as i64);
            }
        });
        
        // Start metrics export if enabled
        if let Some(exporter) = &self.metrics_exporter {
            let exporter = exporter.clone();
            tokio::spawn(async move {
                if let Err(e) = exporter.start_export_server().await {
                    error!("Metrics export server failed: {}", e);
                }
            });
        }
        
        Ok(())
    }
}

impl ClusterMetrics {
    fn new(registry: &Registry) -> Result<Self> {
        let requests_total = Counter::new("spiraldelta_requests_total", "Total number of requests")?;
        let requests_duration = Histogram::new("spiraldelta_request_duration_seconds", "Request duration")?;
        let active_requests = IntGauge::new("spiraldelta_active_requests", "Number of active requests")?;
        
        let nodes_total = IntGauge::new("spiraldelta_nodes_total", "Total number of nodes")?;
        let healthy_nodes = IntGauge::new("spiraldelta_healthy_nodes", "Number of healthy nodes")?;
        let node_cpu_usage = Gauge::new("spiraldelta_node_cpu_usage", "Node CPU usage")?;
        let node_memory_usage = Gauge::new("spiraldelta_node_memory_usage_gb", "Node memory usage in GB")?;
        
        let queries_total = Counter::new("spiraldelta_queries_total", "Total number of queries")?;
        let query_duration = Histogram::new("spiraldelta_query_duration_seconds", "Query duration")?;
        let query_errors = Counter::new("spiraldelta_query_errors_total", "Total query errors")?;
        
        let vectors_total = IntGauge::new("spiraldelta_vectors_total", "Total number of vectors")?;
        let storage_bytes = IntGauge::new("spiraldelta_storage_bytes", "Storage usage in bytes")?;
        let shards_total = IntGauge::new("spiraldelta_shards_total", "Total number of shards")?;
        
        let network_bytes_sent = Counter::new("spiraldelta_network_bytes_sent", "Network bytes sent")?;
        let network_bytes_received = Counter::new("spiraldelta_network_bytes_received", "Network bytes received")?;
        
        // Register all metrics
        registry.register(Box::new(requests_total.clone()))?;
        registry.register(Box::new(requests_duration.clone()))?;
        registry.register(Box::new(active_requests.clone()))?;
        registry.register(Box::new(nodes_total.clone()))?;
        registry.register(Box::new(healthy_nodes.clone()))?;
        registry.register(Box::new(node_cpu_usage.clone()))?;
        registry.register(Box::new(node_memory_usage.clone()))?;
        registry.register(Box::new(queries_total.clone()))?;
        registry.register(Box::new(query_duration.clone()))?;
        registry.register(Box::new(query_errors.clone()))?;
        registry.register(Box::new(vectors_total.clone()))?;
        registry.register(Box::new(storage_bytes.clone()))?;
        registry.register(Box::new(shards_total.clone()))?;
        registry.register(Box::new(network_bytes_sent.clone()))?;
        registry.register(Box::new(network_bytes_received.clone()))?;
        
        Ok(Self {
            requests_total,
            requests_duration,
            active_requests,
            nodes_total,
            healthy_nodes,
            node_cpu_usage,
            node_memory_usage,
            queries_total,
            query_duration,
            query_errors,
            vectors_total,
            storage_bytes,
            shards_total,
            network_bytes_sent,
            network_bytes_received,
        })
    }
}

impl AlertManager {
    async fn new(config: AlertConfig) -> Result<Self> {
        let alert_rules = vec![
            AlertRule {
                id: "high_cpu".to_string(),
                name: "High CPU Usage".to_string(),
                condition: AlertCondition::HighCpuUsage,
                threshold: 0.9,
                level: AlertLevel::Warning,
                cooldown: Duration::from_secs(300), // 5 minutes
                last_triggered: None,
            },
            AlertRule {
                id: "high_memory".to_string(),
                name: "High Memory Usage".to_string(),
                condition: AlertCondition::HighMemoryUsage,
                threshold: 0.9,
                level: AlertLevel::Warning,
                cooldown: Duration::from_secs(300),
                last_triggered: None,
            },
            AlertRule {
                id: "high_error_rate".to_string(),
                name: "High Error Rate".to_string(),
                condition: AlertCondition::HighErrorRate,
                threshold: 0.1, // 10%
                level: AlertLevel::Critical,
                cooldown: Duration::from_secs(60), // 1 minute
                last_triggered: None,
            },
        ];
        
        Ok(Self {
            config,
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_rules,
        })
    }
    
    async fn trigger_alert(
        &self,
        alert_id: String,
        level: AlertLevel,
        title: String,
        description: String,
        node_id: Option<NodeId>,
    ) -> Result<()> {
        let mut active_alerts = self.active_alerts.write().await;
        
        // Check if alert already exists
        if active_alerts.contains_key(&alert_id) {
            return Ok(()); // Don't duplicate alerts
        }
        
        let alert = Alert {
            id: alert_id.clone(),
            level,
            title,
            description,
            node_id,
            created_at: chrono::Utc::now(),
            resolved_at: None,
        };
        
        active_alerts.insert(alert_id.clone(), alert.clone());
        
        // Send alert notification
        self.send_alert_notification(&alert).await?;
        
        info!("Alert triggered: {} - {}", alert.title, alert.description);
        Ok(())
    }
    
    async fn send_alert_notification(&self, alert: &Alert) -> Result<()> {
        if !self.config.enable_alerts {
            return Ok(());
        }
        
        // Send webhook notification if configured
        if let Some(webhook_url) = &self.config.webhook_url {
            // In a real implementation, this would make HTTP POST to webhook
            info!("Sending alert to webhook: {}", webhook_url);
        }
        
        // Send email notification if configured
        if self.config.email_notifications {
            // In a real implementation, this would send email
            info!("Sending alert email notification");
        }
        
        Ok(())
    }
}

impl MetricsExporter {
    async fn new(config: MonitoringConfig, registry: Arc<Registry>) -> Result<Self> {
        Ok(Self {
            config,
            registry,
        })
    }
    
    async fn start_export_server(&self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.config.metrics_port);
        info!("Starting metrics export server on {}", addr);
        
        // In a real implementation, this would start an HTTP server
        // serving Prometheus metrics on /metrics endpoint
        
        // Placeholder for metrics server
        tokio::time::sleep(Duration::from_secs(u64::MAX)).await;
        Ok(())
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enable_alerts: true,
            webhook_url: None,
            email_notifications: false,
            alert_cooldown_seconds: 300,
        }
    }
}

// Add rand for simulation
use rand::Rng;