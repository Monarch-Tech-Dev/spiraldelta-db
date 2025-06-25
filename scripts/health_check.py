#!/usr/bin/env python3
"""
SpiralDelta Health Check Script
Comprehensive health monitoring for all platform components
"""

import asyncio
import aiohttp
import json
import sys
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HealthCheck:
    """Health check result"""
    service: str
    status: str  # healthy, degraded, unhealthy
    response_time: float
    details: Dict[str, Any]
    timestamp: datetime

class SpiralDeltaHealthChecker:
    """Comprehensive health checker for SpiralDelta platform"""
    
    def __init__(self, base_url: str = "http://localhost"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_service_health(self, service_name: str, port: int, path: str = "/health") -> HealthCheck:
        """Check health of a specific service"""
        start_time = time.time()
        url = f"{self.base_url}:{port}{path}"
        
        try:
            async with self.session.get(url) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        status = "healthy"
                        details = data
                    except json.JSONDecodeError:
                        status = "degraded"
                        details = {"error": "Invalid JSON response", "status_code": response.status}
                else:
                    status = "unhealthy"
                    details = {"error": f"HTTP {response.status}", "status_code": response.status}
                    
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            status = "unhealthy"
            details = {"error": "Timeout", "timeout": True}
        except Exception as e:
            response_time = time.time() - start_time
            status = "unhealthy"
            details = {"error": str(e), "exception_type": type(e).__name__}
        
        return HealthCheck(
            service=service_name,
            status=status,
            response_time=response_time,
            details=details,
            timestamp=datetime.now()
        )
    
    async def check_all_services(self) -> List[HealthCheck]:
        """Check health of all SpiralDelta services"""
        services = [
            ("api-aggregator", 5000, "/health"),
            ("sacred-architecture", 5001, "/health"),
            ("grafana", 3000, "/api/health"),
            ("prometheus", 9090, "/-/healthy"),
            ("alertmanager", 9093, "/-/healthy"),
        ]
        
        tasks = [
            self.check_service_health(name, port, path)
            for name, port, path in services
        ]
        
        return await asyncio.gather(*tasks)
    
    async def check_database_health(self) -> HealthCheck:
        """Check database connectivity via API Aggregator"""
        return await self.check_service_health("database", 5000, "/health/db")
    
    async def check_redis_health(self) -> HealthCheck:
        """Check Redis connectivity via API Aggregator"""
        return await self.check_service_health("redis", 5000, "/health/redis")
    
    async def check_api_functionality(self) -> HealthCheck:
        """Test actual API functionality"""
        start_time = time.time()
        
        try:
            # Test API Aggregator functionality
            test_data = {
                "query": "test health check",
                "context": "system monitoring",
                "timestamp": datetime.now().isoformat()
            }
            
            url = f"{self.base_url}:5000/api/v1/query"
            async with self.session.post(url, json=test_data, headers={
                "X-API-Key": "health_check_key",
                "Content-Type": "application/json"
            }) as response:
                response_time = time.time() - start_time
                
                if response.status in [200, 401]:  # 401 is expected for invalid key
                    status = "healthy"
                    details = {
                        "api_responsive": True,
                        "status_code": response.status,
                        "response_time_ms": response_time * 1000
                    }
                else:
                    status = "degraded"
                    details = {
                        "api_responsive": False,
                        "status_code": response.status,
                        "response_time_ms": response_time * 1000
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            status = "unhealthy"
            details = {
                "error": str(e),
                "exception_type": type(e).__name__,
                "response_time_ms": response_time * 1000
            }
        
        return HealthCheck(
            service="api-functionality",
            status=status,
            response_time=response_time,
            details=details,
            timestamp=datetime.now()
        )
    
    async def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive health check of entire platform"""
        logger.info("Starting comprehensive health check...")
        
        # Check all services
        service_checks = await self.check_all_services()
        
        # Check additional components
        additional_checks = await asyncio.gather(
            self.check_database_health(),
            self.check_redis_health(),
            self.check_api_functionality()
        )
        
        all_checks = service_checks + additional_checks
        
        # Analyze overall health
        healthy_count = sum(1 for check in all_checks if check.status == "healthy")
        degraded_count = sum(1 for check in all_checks if check.status == "degraded")
        unhealthy_count = sum(1 for check in all_checks if check.status == "unhealthy")
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Calculate average response time
        avg_response_time = sum(check.response_time for check in all_checks) / len(all_checks)
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_services": len(all_checks),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "average_response_time": avg_response_time
            },
            "service_details": [asdict(check) for check in all_checks],
            "recommendations": self._generate_recommendations(all_checks)
        }
    
    def _generate_recommendations(self, checks: List[HealthCheck]) -> List[str]:
        """Generate recommendations based on health check results"""
        recommendations = []
        
        unhealthy_services = [check.service for check in checks if check.status == "unhealthy"]
        degraded_services = [check.service for check in checks if check.status == "degraded"]
        slow_services = [check.service for check in checks if check.response_time > 1.0]
        
        if unhealthy_services:
            recommendations.append(f"CRITICAL: Restart unhealthy services: {', '.join(unhealthy_services)}")
        
        if degraded_services:
            recommendations.append(f"WARNING: Investigate degraded services: {', '.join(degraded_services)}")
        
        if slow_services:
            recommendations.append(f"PERFORMANCE: Check slow services: {', '.join(slow_services)}")
        
        # Check for specific patterns
        api_check = next((c for c in checks if c.service == "api-aggregator"), None)
        if api_check and api_check.status != "healthy":
            recommendations.append("Check API Aggregator logs: docker-compose logs api-aggregator")
        
        sacred_check = next((c for c in checks if c.service == "sacred-architecture"), None)
        if sacred_check and sacred_check.status != "healthy":
            recommendations.append("Check Sacred Architecture logs: docker-compose logs sacred-architecture")
        
        db_check = next((c for c in checks if c.service == "database"), None)
        if db_check and db_check.status != "healthy":
            recommendations.append("Check database connectivity: docker exec spiraldelta-postgres pg_isready")
        
        if not recommendations:
            recommendations.append("All services are healthy! 🎉")
        
        return recommendations

async def main():
    parser = argparse.ArgumentParser(description="SpiralDelta Health Check")
    parser.add_argument("--base-url", default="http://localhost", help="Base URL for services")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--service", help="Check specific service only")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    async with SpiralDeltaHealthChecker(args.base_url) as checker:
        if args.service:
            # Check specific service
            service_map = {
                "api": ("api-aggregator", 5000, "/health"),
                "sacred": ("sacred-architecture", 5001, "/health"),
                "grafana": ("grafana", 3000, "/api/health"),
                "prometheus": ("prometheus", 9090, "/-/healthy"),
                "alertmanager": ("alertmanager", 9093, "/-/healthy"),
            }
            
            if args.service in service_map:
                name, port, path = service_map[args.service]
                result = await checker.check_service_health(name, port, path)
                
                if args.json:
                    print(json.dumps(asdict(result), indent=2, default=str))
                else:
                    print(f"Service: {result.service}")
                    print(f"Status: {result.status}")
                    print(f"Response Time: {result.response_time:.3f}s")
                    print(f"Details: {result.details}")
            else:
                print(f"Unknown service: {args.service}")
                print(f"Available services: {', '.join(service_map.keys())}")
                sys.exit(1)
        else:
            # Comprehensive check
            result = await checker.run_comprehensive_check()
            
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print("🔍 SpiralDelta Platform Health Check")
                print("=" * 50)
                print(f"Overall Status: {result['overall_status'].upper()}")
                print(f"Timestamp: {result['timestamp']}")
                print()
                
                summary = result['summary']
                print(f"Services Summary:")
                print(f"  Total: {summary['total_services']}")
                print(f"  ✅ Healthy: {summary['healthy']}")
                print(f"  ⚠️  Degraded: {summary['degraded']}")
                print(f"  ❌ Unhealthy: {summary['unhealthy']}")
                print(f"  ⏱️  Avg Response Time: {summary['average_response_time']:.3f}s")
                print()
                
                print("Service Details:")
                for check in result['service_details']:
                    status_icon = "✅" if check['status'] == "healthy" else "⚠️" if check['status'] == "degraded" else "❌"
                    print(f"  {status_icon} {check['service']}: {check['status']} ({check['response_time']:.3f}s)")
                
                print()
                print("Recommendations:")
                for rec in result['recommendations']:
                    print(f"  • {rec}")
            
            # Exit with appropriate code
            if result['overall_status'] == "unhealthy":
                sys.exit(2)
            elif result['overall_status'] == "degraded":
                sys.exit(1)
            else:
                sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nHealth check interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sys.exit(2)