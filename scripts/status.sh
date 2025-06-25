#!/bin/bash
# SpiralDelta Production Status Monitor
# Comprehensive health and status monitoring for production deployment

set -e

# Configuration
HEALTH_CHECK_TIMEOUT=10
DETAILED_OUTPUT="${DETAILED_OUTPUT:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}🚀 SpiralDelta Production Status Monitor${NC}"
    echo "=================================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

check_service_health() {
    local service_name="$1"
    local health_url="$2"
    local expected_response="$3"
    
    if curl -s -f "$health_url" --max-time "$HEALTH_CHECK_TIMEOUT" | grep -q "$expected_response" 2>/dev/null; then
        echo -e "✅ ${GREEN}$service_name${NC} - Healthy"
        return 0
    else
        echo -e "❌ ${RED}$service_name${NC} - Unhealthy"
        return 1
    fi
}

check_container_status() {
    local container_name="$1"
    
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$container_name.*Up"; then
        status=$(docker ps --format "{{.Status}}" --filter "name=$container_name")
        echo -e "🐳 ${GREEN}$container_name${NC} - $status"
        return 0
    else
        echo -e "🐳 ${RED}$container_name${NC} - Not running"
        return 1
    fi
}

check_database_connectivity() {
    if docker exec spiraldelta-postgres pg_isready -U spiraldelta >/dev/null 2>&1; then
        echo -e "🗄️  ${GREEN}PostgreSQL${NC} - Connected"
        
        if [ "$DETAILED_OUTPUT" = "true" ]; then
            db_size=$(docker exec spiraldelta-postgres psql -U spiraldelta -t -c "SELECT pg_size_pretty(pg_database_size('spiraldelta'));" | tr -d ' ')
            echo "   📊 Database size: $db_size"
            
            active_connections=$(docker exec spiraldelta-postgres psql -U spiraldelta -t -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';" | tr -d ' ')
            echo "   🔗 Active connections: $active_connections"
        fi
        return 0
    else
        echo -e "🗄️  ${RED}PostgreSQL${NC} - Connection failed"
        return 1
    fi
}

check_redis_connectivity() {
    if docker exec spiraldelta-redis redis-cli ping | grep -q PONG 2>/dev/null; then
        echo -e "🔴 ${GREEN}Redis${NC} - Connected"
        
        if [ "$DETAILED_OUTPUT" = "true" ]; then
            memory_usage=$(docker exec spiraldelta-redis redis-cli info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
            echo "   💾 Memory usage: $memory_usage"
            
            connected_clients=$(docker exec spiraldelta-redis redis-cli info clients | grep connected_clients | cut -d: -f2 | tr -d '\r')
            echo "   👥 Connected clients: $connected_clients"
        fi
        return 0
    else
        echo -e "🔴 ${RED}Redis${NC} - Connection failed"
        return 1
    fi
}

check_disk_space() {
    echo -e "${BLUE}💾 Disk Space Usage:${NC}"
    
    # Check system disk space
    df -h / | tail -1 | while read filesystem size used avail percent mountpoint; do
        usage_num=$(echo "$percent" | sed 's/%//')
        if [ "$usage_num" -gt 90 ]; then
            echo -e "   ⚠️  ${RED}Root filesystem: $percent used${NC}"
        elif [ "$usage_num" -gt 80 ]; then
            echo -e "   ⚠️  ${YELLOW}Root filesystem: $percent used${NC}"
        else
            echo -e "   ✅ ${GREEN}Root filesystem: $percent used${NC}"
        fi
    done
    
    # Check Docker volumes
    if [ "$DETAILED_OUTPUT" = "true" ]; then
        echo "   Docker volumes:"
        docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}" | grep -v TYPE
    fi
}

check_resource_usage() {
    echo -e "${BLUE}🔧 Resource Usage:${NC}"
    
    # Memory usage
    memory_info=$(free -h | grep '^Mem:')
    echo "   $memory_info"
    
    # CPU load
    load_avg=$(uptime | awk -F'load average:' '{print $2}')
    echo "   Load average:$load_avg"
    
    if [ "$DETAILED_OUTPUT" = "true" ]; then
        echo -e "\n   ${BLUE}Container Resource Usage:${NC}"
        docker stats --no-stream --format "   {{.Name}}: CPU {{.CPUPerc}}, Memory {{.MemUsage}}"
    fi
}

check_service_logs() {
    echo -e "${BLUE}📋 Recent Service Logs:${NC}"
    
    # Check for recent errors in logs
    error_count=$(docker-compose logs --tail=100 2>/dev/null | grep -i error | wc -l)
    warning_count=$(docker-compose logs --tail=100 2>/dev/null | grep -i warning | wc -l)
    
    if [ "$error_count" -gt 0 ]; then
        echo -e "   ⚠️  ${RED}$error_count recent errors found${NC}"
    else
        echo -e "   ✅ ${GREEN}No recent errors${NC}"
    fi
    
    if [ "$warning_count" -gt 0 ]; then
        echo -e "   ⚠️  ${YELLOW}$warning_count recent warnings found${NC}"
    else
        echo -e "   ✅ ${GREEN}No recent warnings${NC}"
    fi
    
    if [ "$DETAILED_OUTPUT" = "true" ] && [ "$error_count" -gt 0 ]; then
        echo -e "\n   ${RED}Recent Errors:${NC}"
        docker-compose logs --tail=20 2>/dev/null | grep -i error | tail -5 | sed 's/^/   /'
    fi
}

check_ssl_certificates() {
    echo -e "${BLUE}🔒 SSL Certificate Status:${NC}"
    
    # Check if Traefik ACME file exists
    if docker exec spiraldelta-traefik test -f /acme.json 2>/dev/null; then
        echo -e "   ✅ ${GREEN}ACME certificate file exists${NC}"
        
        # Check certificate expiry (simplified check)
        # In production, you'd want a more sophisticated certificate monitoring
        echo -e "   ℹ️  ${BLUE}Certificate monitoring available via Traefik dashboard${NC}"
    else
        echo -e "   ⚠️  ${YELLOW}ACME certificate file not found${NC}"
    fi
}

check_backup_status() {
    echo -e "${BLUE}💾 Backup Status:${NC}"
    
    backup_dir="/opt/spiraldelta-backups"
    if [ -d "$backup_dir" ]; then
        latest_backup=$(find "$backup_dir" -maxdepth 1 -type d -name "*_*" | sort -r | head -n 1)
        if [ -n "$latest_backup" ]; then
            backup_date=$(basename "$latest_backup")
            echo -e "   ✅ ${GREEN}Latest backup: $backup_date${NC}"
            
            # Check backup age
            backup_timestamp=$(echo "$backup_date" | sed 's/_/ /' | sed 's/\([0-9]\{8\}\) \([0-9]\{6\}\)/\1 \2/')
            backup_epoch=$(date -d "$backup_timestamp" +%s 2>/dev/null || echo "0")
            current_epoch=$(date +%s)
            age_hours=$(( (current_epoch - backup_epoch) / 3600 ))
            
            if [ "$age_hours" -gt 24 ]; then
                echo -e "   ⚠️  ${YELLOW}Backup is $age_hours hours old${NC}"
            else
                echo -e "   ✅ ${GREEN}Backup is recent ($age_hours hours old)${NC}"
            fi
        else
            echo -e "   ❌ ${RED}No backups found${NC}"
        fi
    else
        echo -e "   ❌ ${RED}Backup directory not found${NC}"
    fi
}

show_service_urls() {
    echo -e "${BLUE}🌐 Service URLs:${NC}"
    echo "   API Aggregator: http://localhost:5000"
    echo "   Sacred Architecture: http://localhost:5001"
    echo "   Grafana Dashboard: http://localhost:3000"
    echo "   Prometheus: http://localhost:9090"
    echo "   Traefik Dashboard: http://localhost:8080"
}

main() {
    print_header
    
    # Overall health status
    overall_status=0
    
    echo -e "${BLUE}🔍 Service Health Checks:${NC}"
    check_service_health "API Aggregator" "http://localhost:5000/health" "healthy" || overall_status=1
    check_service_health "Sacred Architecture" "http://localhost:5001/health" "healthy" || overall_status=1
    
    echo -e "\n${BLUE}🐳 Container Status:${NC}"
    check_container_status "spiraldelta-api-aggregator" || overall_status=1
    check_container_status "spiraldelta-sacred" || overall_status=1
    check_container_status "spiraldelta-postgres" || overall_status=1
    check_container_status "spiraldelta-redis" || overall_status=1
    check_container_status "spiraldelta-traefik" || overall_status=1
    
    echo -e "\n${BLUE}🔌 Database Connectivity:${NC}"
    check_database_connectivity || overall_status=1
    check_redis_connectivity || overall_status=1
    
    echo ""
    check_disk_space
    
    echo ""
    check_resource_usage
    
    echo ""
    check_service_logs
    
    echo ""
    check_ssl_certificates
    
    echo ""
    check_backup_status
    
    echo ""
    show_service_urls
    
    # Overall status summary
    echo ""
    echo "=================================================="
    if [ $overall_status -eq 0 ]; then
        echo -e "🎉 ${GREEN}Overall Status: HEALTHY${NC}"
    else
        echo -e "⚠️  ${YELLOW}Overall Status: ISSUES DETECTED${NC}"
        echo "   Run with DETAILED_OUTPUT=true for more information"
        echo "   Check logs: docker-compose logs"
    fi
    
    return $overall_status
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--detailed)
            DETAILED_OUTPUT="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --detailed    Show detailed output"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main status check
main