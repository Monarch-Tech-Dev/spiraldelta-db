/*!
Distributed Gateway

Client-facing API gateway with load balancing and request routing.
*/

use crate::*;
use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug, instrument};
use serde::{Serialize, Deserialize};
use hyper::{Body, Request, Response, StatusCode};
use tower::{Service, ServiceBuilder};
use tower_http::cors::CorsLayer;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// API Gateway for distributed SpiralDelta cluster
pub struct ApiGateway {
    /// Gateway configuration
    config: GatewayConfig,
    
    /// Cluster topology reference
    topology: Arc<RwLock<ClusterTopology>>,
    
    /// Load balancer for request routing
    load_balancer: Arc<LoadBalancer>,
    
    /// Request router
    router: Arc<RequestRouter>,
    
    /// Authentication service
    auth_service: Arc<AuthService>,
    
    /// Rate limiter
    rate_limiter: Arc<RateLimiter>,
    
    /// Monitoring system
    monitoring: Arc<MonitoringSystem>,
    
    /// Active connections tracking
    active_connections: Arc<RwLock<HashMap<String, ConnectionInfo>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Gateway listening address
    pub bind_address: std::net::SocketAddr,
    
    /// Maximum concurrent connections
    pub max_connections: usize,
    
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,
    
    /// Enable authentication
    pub enable_auth: bool,
    
    /// Rate limiting configuration
    pub rate_limiting: RateLimitConfig,
    
    /// CORS settings
    pub cors: CorsConfig,
    
    /// Circuit breaker settings
    pub circuit_breaker: CircuitBreakerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per second per client
    pub requests_per_second: u32,
    
    /// Burst capacity
    pub burst_size: u32,
    
    /// Rate limit window in seconds
    pub window_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorsConfig {
    /// Allow credentials
    pub allow_credentials: bool,
    
    /// Allowed origins
    pub allowed_origins: Vec<String>,
    
    /// Allowed methods
    pub allowed_methods: Vec<String>,
    
    /// Allowed headers
    pub allowed_headers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold before opening circuit
    pub failure_threshold: u32,
    
    /// Success threshold to close circuit
    pub success_threshold: u32,
    
    /// Timeout before trying to close circuit (seconds)
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone)]
struct ConnectionInfo {
    client_id: String,
    connected_at: chrono::DateTime<chrono::Utc>,
    request_count: u64,
    last_activity: chrono::DateTime<chrono::Utc>,
}

/// Request router for distributing queries to appropriate nodes
pub struct RequestRouter {
    config: GatewayConfig,
    topology: Arc<RwLock<ClusterTopology>>,
    load_balancer: Arc<LoadBalancer>,
    monitoring: Arc<MonitoringSystem>,
}

/// Authentication service for API access control
pub struct AuthService {
    config: GatewayConfig,
    api_keys: Arc<RwLock<HashMap<String, ApiKeyInfo>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiKeyInfo {
    key_id: String,
    client_name: String,
    permissions: Vec<Permission>,
    rate_limit: Option<RateLimitConfig>,
    created_at: chrono::DateTime<chrono::Utc>,
    last_used: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Permission {
    Read,
    Write,
    Admin,
    Metrics,
}

/// Rate limiter for controlling request rates
pub struct RateLimiter {
    config: RateLimitConfig,
    client_buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
}

#[derive(Debug)]
struct TokenBucket {
    tokens: f64,
    last_refill: chrono::DateTime<chrono::Utc>,
    capacity: f64,
    refill_rate: f64,
}

impl ApiGateway {
    /// Create a new API gateway
    pub async fn new(
        config: GatewayConfig,
        topology: Arc<RwLock<ClusterTopology>>,
        load_balancer: Arc<LoadBalancer>,
        monitoring: Arc<MonitoringSystem>,
    ) -> Result<Self> {
        let router = Arc::new(RequestRouter::new(
            config.clone(),
            topology.clone(),
            load_balancer.clone(),
            monitoring.clone(),
        ));
        
        let auth_service = Arc::new(AuthService::new(config.clone()).await?);
        let rate_limiter = Arc::new(RateLimiter::new(config.rate_limiting.clone()));
        
        Ok(Self {
            config,
            topology,
            load_balancer,
            router,
            auth_service,
            rate_limiter,
            monitoring,
            active_connections: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Start the API gateway server
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting API gateway on {}", self.config.bind_address);
        
        // Create HTTP service with middleware stack
        let service = ServiceBuilder::new()
            .layer(CorsLayer::permissive()) // Add CORS support
            .service(GatewayService::new(
                self.router.clone(),
                self.auth_service.clone(),
                self.rate_limiter.clone(),
                self.monitoring.clone(),
                self.active_connections.clone(),
            ));
        
        // Start HTTP server
        let server = hyper::Server::bind(&self.config.bind_address)
            .http1_preserve_header_case(true)
            .http1_title_case_headers(true)
            .serve(tower::make::Shared::new(service));
        
        // Start background tasks
        self.start_background_tasks().await?;
        
        info!("API gateway started successfully on {}", self.config.bind_address);
        
        // Run server
        if let Err(e) = server.await {
            error!("API gateway server error: {}", e);
            return Err(anyhow!("Server error: {}", e));
        }
        
        Ok(())
    }
    
    async fn start_background_tasks(&self) -> Result<()> {
        // Start connection cleanup task
        let active_connections = self.active_connections.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                
                let mut connections = active_connections.write().await;
                let now = chrono::Utc::now();
                
                // Remove stale connections (inactive for 5 minutes)
                connections.retain(|_, conn| {
                    (now - conn.last_activity).num_seconds() < 300
                });
            }
        });
        
        // Start metrics collection
        let monitoring = self.monitoring.clone();
        let active_connections = self.active_connections.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                
                let connections = active_connections.read().await;
                let connection_count = connections.len();
                
                if let Err(e) = monitoring.record_gateway_metrics(connection_count).await {
                    warn!("Failed to record gateway metrics: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Get gateway statistics
    pub async fn get_statistics(&self) -> Result<GatewayStatistics> {
        let active_connections = self.active_connections.read().await;
        
        let total_requests: u64 = active_connections.values()
            .map(|conn| conn.request_count)
            .sum();
        
        Ok(GatewayStatistics {
            active_connections: active_connections.len(),
            total_requests,
            requests_per_second: self.calculate_rps().await?,
            avg_response_time_ms: self.monitoring.get_avg_response_time().await?,
            error_rate: self.monitoring.get_error_rate().await?,
        })
    }
    
    async fn calculate_rps(&self) -> Result<f64> {
        // Placeholder for RPS calculation
        Ok(100.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayStatistics {
    pub active_connections: usize,
    pub total_requests: u64,
    pub requests_per_second: f64,
    pub avg_response_time_ms: f64,
    pub error_rate: f64,
}

/// HTTP service implementation for the gateway
struct GatewayService {
    router: Arc<RequestRouter>,
    auth_service: Arc<AuthService>,
    rate_limiter: Arc<RateLimiter>,
    monitoring: Arc<MonitoringSystem>,
    active_connections: Arc<RwLock<HashMap<String, ConnectionInfo>>>,
}

impl GatewayService {
    fn new(
        router: Arc<RequestRouter>,
        auth_service: Arc<AuthService>,
        rate_limiter: Arc<RateLimiter>,
        monitoring: Arc<MonitoringSystem>,
        active_connections: Arc<RwLock<HashMap<String, ConnectionInfo>>>,
    ) -> Self {
        Self {
            router,
            auth_service,
            rate_limiter,
            monitoring,
            active_connections,
        }
    }
    
    async fn handle_request(&self, req: Request<Body>) -> Result<Response<Body>, Infallible> {
        let start_time = std::time::Instant::now();
        
        // Extract client information
        let client_ip = req.headers()
            .get("x-forwarded-for")
            .or_else(|| req.headers().get("x-real-ip"))
            .and_then(|h| h.to_str().ok())
            .unwrap_or("unknown")
            .to_string();
        
        // Update connection tracking
        self.update_connection_info(&client_ip).await;
        
        // Handle the request
        let response = match self.process_request(req).await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Request processing failed: {}", e);
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::from(format!("Internal server error: {}", e)))
                    .unwrap_or_else(|_| Response::new(Body::empty()))
            }
        };
        
        // Record metrics
        let duration = start_time.elapsed().as_millis() as u64;
        if let Err(e) = self.monitoring.record_request_metrics(duration, response.status().as_u16()).await {
            warn!("Failed to record request metrics: {}", e);
        }
        
        Ok(response)
    }
    
    async fn process_request(&self, req: Request<Body>) -> Result<Response<Body>> {
        let method = req.method().clone();
        let path = req.uri().path();
        
        // Extract client ID for rate limiting and authentication
        let client_id = self.extract_client_id(&req)?;
        
        // Check rate limits
        if !self.rate_limiter.check_rate_limit(&client_id).await? {
            return Ok(Response::builder()
                .status(StatusCode::TOO_MANY_REQUESTS)
                .body(Body::from("Rate limit exceeded"))?);
        }
        
        // Authenticate request if enabled
        if self.auth_service.config.enable_auth {
            if let Err(e) = self.auth_service.authenticate(&req).await {
                return Ok(Response::builder()
                    .status(StatusCode::UNAUTHORIZED)
                    .body(Body::from(format!("Authentication failed: {}", e)))?);
            }
        }
        
        // Route the request
        match (method.as_str(), path) {
            ("POST", "/api/v1/search") => self.handle_similarity_search(req).await,
            ("POST", "/api/v1/insert") => self.handle_vector_insertion(req).await,
            ("POST", "/api/v1/batch") => self.handle_batch_operations(req).await,
            ("GET", "/api/v1/cluster/status") => self.handle_cluster_status(req).await,
            ("GET", "/api/v1/cluster/stats") => self.handle_cluster_stats(req).await,
            ("GET", "/health") => self.handle_health_check(req).await,
            ("GET", "/metrics") => self.handle_metrics(req).await,
            _ => Ok(Response::builder()
                .status(StatusCode::NOT_FOUND)
                .body(Body::from("Endpoint not found"))?),
        }
    }
    
    async fn handle_similarity_search(&self, req: Request<Body>) -> Result<Response<Body>> {
        // Parse request body
        let body_bytes = hyper::body::to_bytes(req.into_body()).await?;
        let search_request: SimilaritySearchRequest = serde_json::from_slice(&body_bytes)?;
        
        // Create distributed query
        let query = DistributedQuery {
            query_id: uuid::Uuid::new_v4().to_string(),
            original_query: search_request.vector,
            query_type: QueryType::SimilaritySearch,
            parameters: QueryParameters {
                k: Some(search_request.k),
                metric: search_request.metric,
                radius: search_request.radius,
                extra_params: search_request.extra_params,
            },
            target_shards: vec![], // Will be determined by router
            execution_plan: ExecutionPlan {
                steps: vec![],
                estimated_duration_ms: 0,
                resource_requirements: ResourceRequirements {
                    memory_gb: 0.1,
                    cpu_cores: 0.1,
                    network_mbps: 10.0,
                    gpu_required: false,
                },
            },
            priority: QueryPriority::Normal,
            deadline: Some(chrono::Utc::now() + chrono::Duration::seconds(30)),
        };
        
        // Route query to appropriate nodes
        let result = self.router.route_query(query).await?;
        
        // Format response
        let response_body = serde_json::to_string(&result)?;
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(response_body))?)
    }
    
    async fn handle_vector_insertion(&self, req: Request<Body>) -> Result<Response<Body>> {
        let body_bytes = hyper::body::to_bytes(req.into_body()).await?;
        let insert_request: VectorInsertRequest = serde_json::from_slice(&body_bytes)?;
        
        let query = DistributedQuery {
            query_id: uuid::Uuid::new_v4().to_string(),
            original_query: insert_request.vector,
            query_type: QueryType::VectorInsertion,
            parameters: QueryParameters {
                k: None,
                metric: None,
                radius: None,
                extra_params: insert_request.metadata,
            },
            target_shards: vec![],
            execution_plan: ExecutionPlan {
                steps: vec![],
                estimated_duration_ms: 0,
                resource_requirements: ResourceRequirements {
                    memory_gb: 0.05,
                    cpu_cores: 0.05,
                    network_mbps: 5.0,
                    gpu_required: false,
                },
            },
            priority: QueryPriority::Normal,
            deadline: Some(chrono::Utc::now() + chrono::Duration::seconds(10)),
        };
        
        let result = self.router.route_query(query).await?;
        
        let response_body = serde_json::to_string(&result)?;
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(response_body))?)
    }
    
    async fn handle_batch_operations(&self, _req: Request<Body>) -> Result<Response<Body>> {
        // Placeholder for batch operations
        Ok(Response::builder()
            .status(StatusCode::OK)
            .body(Body::from("Batch operations not yet implemented"))?)
    }
    
    async fn handle_cluster_status(&self, _req: Request<Body>) -> Result<Response<Body>> {
        let query = DistributedQuery {
            query_id: uuid::Uuid::new_v4().to_string(),
            original_query: vec![],
            query_type: QueryType::ClusterStatus,
            parameters: QueryParameters {
                k: None,
                metric: None,
                radius: None,
                extra_params: HashMap::new(),
            },
            target_shards: vec![],
            execution_plan: ExecutionPlan {
                steps: vec![],
                estimated_duration_ms: 0,
                resource_requirements: ResourceRequirements {
                    memory_gb: 0.01,
                    cpu_cores: 0.01,
                    network_mbps: 1.0,
                    gpu_required: false,
                },
            },
            priority: QueryPriority::High,
            deadline: Some(chrono::Utc::now() + chrono::Duration::seconds(5)),
        };
        
        let result = self.router.route_query(query).await?;
        
        let response_body = serde_json::to_string(&result)?;
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(response_body))?)
    }
    
    async fn handle_cluster_stats(&self, _req: Request<Body>) -> Result<Response<Body>> {
        // Placeholder for cluster statistics
        let stats = serde_json::json!({
            "status": "healthy",
            "nodes": 3,
            "vectors": 1000000,
            "queries_per_second": 100.0
        });
        
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(stats.to_string()))?)
    }
    
    async fn handle_health_check(&self, _req: Request<Body>) -> Result<Response<Body>> {
        Ok(Response::builder()
            .status(StatusCode::OK)
            .body(Body::from("OK"))?)
    }
    
    async fn handle_metrics(&self, _req: Request<Body>) -> Result<Response<Body>> {
        // Placeholder for Prometheus metrics
        let metrics = "# HELP spiraldelta_requests_total Total requests\n# TYPE spiraldelta_requests_total counter\nspiraledelta_requests_total 1000\n";
        
        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/plain")
            .body(Body::from(metrics))?)
    }
    
    fn extract_client_id(&self, req: &Request<Body>) -> Result<String> {
        // Extract client ID from API key or IP address
        if let Some(auth_header) = req.headers().get("authorization") {
            if let Ok(auth_str) = auth_header.to_str() {
                if auth_str.starts_with("Bearer ") {
                    return Ok(auth_str[7..].to_string());
                }
            }
        }
        
        // Fallback to IP address
        Ok(req.headers()
            .get("x-forwarded-for")
            .or_else(|| req.headers().get("x-real-ip"))
            .and_then(|h| h.to_str().ok())
            .unwrap_or("unknown")
            .to_string())
    }
    
    async fn update_connection_info(&self, client_id: &str) {
        let mut connections = self.active_connections.write().await;
        let now = chrono::Utc::now();
        
        connections.entry(client_id.to_string())
            .and_modify(|conn| {
                conn.request_count += 1;
                conn.last_activity = now;
            })
            .or_insert(ConnectionInfo {
                client_id: client_id.to_string(),
                connected_at: now,
                request_count: 1,
                last_activity: now,
            });
    }
}

// Implement Tower Service trait for GatewayService
impl<ReqBody> Service<Request<ReqBody>> for GatewayService 
where
    ReqBody: Into<Body>,
{
    type Response = Response<Body>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
    
    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
    
    fn call(&mut self, req: Request<ReqBody>) -> Self::Future {
        let req = req.map(Into::into);
        let service = self.clone();
        
        Box::pin(async move {
            service.handle_request(req).await
        })
    }
}

impl Clone for GatewayService {
    fn clone(&self) -> Self {
        Self {
            router: self.router.clone(),
            auth_service: self.auth_service.clone(),
            rate_limiter: self.rate_limiter.clone(),
            monitoring: self.monitoring.clone(),
            active_connections: self.active_connections.clone(),
        }
    }
}

// Request/Response types
#[derive(Debug, Serialize, Deserialize)]
struct SimilaritySearchRequest {
    vector: Vec<f32>,
    k: usize,
    metric: Option<String>,
    radius: Option<f64>,
    extra_params: HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VectorInsertRequest {
    vector: Vec<f32>,
    metadata: HashMap<String, String>,
}

impl RequestRouter {
    pub fn new(
        config: GatewayConfig,
        topology: Arc<RwLock<ClusterTopology>>,
        load_balancer: Arc<LoadBalancer>,
        monitoring: Arc<MonitoringSystem>,
    ) -> Self {
        Self {
            config,
            topology,
            load_balancer,
            monitoring,
        }
    }
    
    pub async fn route_query(&self, query: DistributedQuery) -> Result<QueryResult> {
        // Find appropriate coordinator node
        let topology = self.topology.read().await;
        let coordinator_nodes: Vec<_> = topology.nodes.values()
            .filter(|node| matches!(node.node_type, NodeType::Coordinator) && matches!(node.status, NodeStatus::Healthy))
            .collect();
        
        if coordinator_nodes.is_empty() {
            return Err(anyhow!("No healthy coordinator nodes available"));
        }
        
        // Use load balancer to select coordinator
        let selected_node = self.load_balancer.select_node(&coordinator_nodes).await?;
        
        // Forward query to coordinator (placeholder)
        // In real implementation, this would make gRPC/HTTP call
        Ok(QueryResult {
            query_id: query.query_id,
            status: QueryStatus::Completed,
            results: QueryResultData::SimilarityResults(vec![(0, 1.0), (1, 0.9)]),
            statistics: QueryStatistics {
                total_duration_ms: 50,
                nodes_involved: 1,
                shards_queried: 1,
                network_round_trips: 1,
                data_transferred_bytes: 1024,
                per_node_stats: HashMap::new(),
            },
            error: None,
        })
    }
}

impl AuthService {
    pub async fn new(config: GatewayConfig) -> Result<Self> {
        let api_keys = Arc::new(RwLock::new(HashMap::new()));
        
        // Initialize with default API keys (in production, these would be loaded from secure storage)
        if config.enable_auth {
            let mut keys = api_keys.write().await;
            keys.insert("demo-key-123".to_string(), ApiKeyInfo {
                key_id: "demo-key-123".to_string(),
                client_name: "Demo Client".to_string(),
                permissions: vec![Permission::Read, Permission::Write],
                rate_limit: None,
                created_at: chrono::Utc::now(),
                last_used: None,
            });
        }
        
        Ok(Self {
            config,
            api_keys,
        })
    }
    
    pub async fn authenticate(&self, req: &Request<Body>) -> Result<()> {
        if !self.config.enable_auth {
            return Ok(());
        }
        
        let auth_header = req.headers().get("authorization")
            .ok_or_else(|| anyhow!("Missing authorization header"))?;
        
        let auth_str = auth_header.to_str()
            .map_err(|_| anyhow!("Invalid authorization header"))?;
        
        if !auth_str.starts_with("Bearer ") {
            return Err(anyhow!("Invalid authorization format"));
        }
        
        let api_key = &auth_str[7..];
        let api_keys = self.api_keys.read().await;
        
        if !api_keys.contains_key(api_key) {
            return Err(anyhow!("Invalid API key"));
        }
        
        // Update last used timestamp
        drop(api_keys);
        let mut api_keys = self.api_keys.write().await;
        if let Some(key_info) = api_keys.get_mut(api_key) {
            key_info.last_used = Some(chrono::Utc::now());
        }
        
        Ok(())
    }
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            client_buckets: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn check_rate_limit(&self, client_id: &str) -> Result<bool> {
        let mut buckets = self.client_buckets.write().await;
        let now = chrono::Utc::now();
        
        let bucket = buckets.entry(client_id.to_string())
            .or_insert_with(|| TokenBucket {
                tokens: self.config.requests_per_second as f64,
                last_refill: now,
                capacity: self.config.burst_size as f64,
                refill_rate: self.config.requests_per_second as f64,
            });
        
        // Refill tokens based on elapsed time
        let elapsed = (now - bucket.last_refill).num_milliseconds() as f64 / 1000.0;
        let tokens_to_add = elapsed * bucket.refill_rate;
        bucket.tokens = (bucket.tokens + tokens_to_add).min(bucket.capacity);
        bucket.last_refill = now;
        
        // Check if request is allowed
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:8080".parse().unwrap(),
            max_connections: 10000,
            request_timeout_ms: 30000,
            enable_auth: false,
            rate_limiting: RateLimitConfig {
                requests_per_second: 100,
                burst_size: 200,
                window_seconds: 60,
            },
            cors: CorsConfig {
                allow_credentials: true,
                allowed_origins: vec!["*".to_string()],
                allowed_methods: vec!["GET".to_string(), "POST".to_string(), "PUT".to_string(), "DELETE".to_string()],
                allowed_headers: vec!["*".to_string()],
            },
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: 10,
                success_threshold: 5,
                timeout_seconds: 60,
            },
        }
    }
}