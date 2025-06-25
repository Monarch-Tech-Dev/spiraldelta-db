#!/bin/bash
# Build GPU acceleration module for SpiralDelta

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Building SpiralDelta GPU Acceleration${NC}"
echo "=================================================="

# Check for CUDA
check_cuda() {
    echo "Checking CUDA availability..."
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9.]*\).*/\1/')
        echo -e "✅ ${GREEN}CUDA ${CUDA_VERSION} found${NC}"
    else
        echo -e "⚠️  ${YELLOW}CUDA not found. GPU acceleration will be disabled.${NC}"
        return 1
    fi
    
    # Check for compatible GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        echo -e "🎮 GPU: ${GPU_INFO}"
    else
        echo -e "⚠️  ${YELLOW}nvidia-smi not found. Cannot detect GPU.${NC}"
    fi
}

# Check Rust toolchain
check_rust() {
    echo "Checking Rust toolchain..."
    
    if command -v rustc &> /dev/null; then
        RUST_VERSION=$(rustc --version | awk '{print $2}')
        echo -e "✅ ${GREEN}Rust ${RUST_VERSION} found${NC}"
    else
        echo -e "❌ ${RED}Rust not found. Please install Rust from https://rustup.rs/${NC}"
        exit 1
    fi
    
    # Check for required targets
    if rustup target list --installed | grep -q "x86_64-unknown-linux-gnu"; then
        echo -e "✅ ${GREEN}Required Rust target available${NC}"
    else
        echo "Installing required Rust target..."
        rustup target add x86_64-unknown-linux-gnu
    fi
}

# Build GPU module
build_gpu_module() {
    echo "Building GPU acceleration module..."
    
    cd spiraldelta-gpu
    
    # Set environment variables for CUDA
    if check_cuda; then
        export CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
        export PATH=${CUDA_ROOT}/bin:${PATH}
        export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}
        
        echo "Building with CUDA support..."
        cargo build --release --features cuda
    else
        echo "Building without CUDA support..."
        cargo build --release --no-default-features
    fi
    
    cd ..
}

# Build Python bindings
build_python_bindings() {
    echo "Building Python bindings..."
    
    # Check for maturin
    if ! command -v maturin &> /dev/null; then
        echo "Installing maturin..."
        pip install maturin
    fi
    
    cd spiraldelta-gpu
    
    # Build Python wheel
    if check_cuda > /dev/null 2>&1; then
        echo "Building Python wheel with CUDA support..."
        maturin build --release --features cuda
    else
        echo "Building Python wheel without CUDA support..."
        maturin build --release --no-default-features
    fi
    
    # Install the wheel
    echo "Installing Python wheel..."
    pip install target/wheels/*.whl --force-reinstall
    
    cd ..
}

# Run tests
run_tests() {
    echo "Running GPU acceleration tests..."
    
    cd spiraldelta-gpu
    
    # Run Rust tests
    echo "Running Rust tests..."
    cargo test --release
    
    # Run Python integration tests
    echo "Running Python integration tests..."
    python -c "
import spiraldelta_gpu
print('✅ GPU module import successful')

if spiraldelta_gpu.check_cuda_availability():
    print('✅ CUDA available')
    memory_info = spiraldelta_gpu.get_gpu_memory_info()
    print(f'🎮 GPU Memory: {memory_info[\"total_gb\"]:.1f} GB total, {memory_info[\"available_gb\"]:.1f} GB available')
else:
    print('⚠️  CUDA not available - CPU fallback will be used')
"
    
    cd ..
}

# Benchmark performance
run_benchmarks() {
    echo "Running performance benchmarks..."
    
    python -c "
import numpy as np
from spiraldelta.gpu_acceleration import get_gpu_engine
import time

print('🔥 Running GPU vs CPU performance benchmark...')

engine = get_gpu_engine()
print(f'GPU Available: {engine.is_gpu_available()}')

# Small benchmark
queries = np.random.randn(100, 128).astype(np.float32)
index = np.random.randn(1000, 128).astype(np.float32)

# GPU benchmark
if engine.is_gpu_available():
    start = time.time()
    gpu_results = engine.similarity_search(queries, index, k=10)
    gpu_time = time.time() - start
    print(f'GPU Time: {gpu_time:.3f}s')
else:
    print('GPU not available for benchmarking')

# CPU benchmark
start = time.time()
cpu_results = engine._cpu_similarity_search(queries, index, 10, 'cosine')
cpu_time = time.time() - start
print(f'CPU Time: {cpu_time:.3f}s')

if engine.is_gpu_available():
    speedup = cpu_time / gpu_time
    print(f'🚀 GPU Speedup: {speedup:.1f}x')

print('✅ Benchmark completed successfully')
"
}

# Update requirements
update_requirements() {
    echo "Updating requirements for GPU acceleration..."
    
    if check_cuda > /dev/null 2>&1; then
        # Add CUDA requirements
        cat >> requirements.txt << EOF

# GPU Acceleration (CUDA)
cupy-cuda12x>=12.0.0
cudf>=23.0.0
EOF
    fi
    
    # Add Rust build requirements
    cat >> requirements-dev.txt << EOF

# GPU Module Build Requirements
maturin>=1.0.0
cffi>=1.15.0
EOF
}

# Main execution
main() {
    echo "Starting GPU acceleration build process..."
    
    # Change to project root
    cd "$(dirname "$0")/.."
    
    # Run build steps
    check_rust
    build_gpu_module
    build_python_bindings
    run_tests
    update_requirements
    run_benchmarks
    
    echo ""
    echo -e "${GREEN}🎉 GPU acceleration build completed successfully!${NC}"
    echo ""
    echo "GPU acceleration is now available in SpiralDelta:"
    echo "  from spiraldelta import check_gpu_availability, gpu_similarity_search"
    echo "  print(f'GPU Available: {check_gpu_availability()}')"
    echo ""
    echo "For detailed usage examples, see:"
    echo "  - src/spiraldelta/gpu_acceleration.py"
    echo "  - spiraldelta-gpu/examples/"
    echo ""
}

# Handle script interruption
trap 'echo -e "\n${RED}Build interrupted${NC}"; exit 1' INT

# Run main function
main "$@"