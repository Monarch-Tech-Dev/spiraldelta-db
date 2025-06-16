# Contributing to SpiralDeltaDB

Thank you for your interest in contributing to SpiralDeltaDB! This guide will help you get started with contributing to our geometric vector database project.

## 🚀 Quick Start for Contributors

### Development Environment Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR-USERNAME/spiraldelta-db
   cd spiraldelta-db
   ```

2. **Set up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate   # Windows
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   pip install -r requirements-dev.txt
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

### Verify Your Setup

```bash
# Run tests
python run_tests.py

# Run examples
python examples/basic_usage.py

# Run benchmarks
python scripts/benchmark.py
```

## 🎯 How to Contribute

### Issues and Bug Reports

1. **Check existing issues** before creating new ones
2. **Use issue templates** when reporting bugs
3. **Include reproduction steps** and environment details
4. **Label appropriately**: bug, enhancement, documentation, etc.

### Feature Requests

1. **Open a discussion first** for major features
2. **Explain the use case** and potential impact
3. **Consider implementation complexity**
4. **Get maintainer feedback** before starting work

### Pull Requests

1. **Create feature branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make focused changes** - one feature per PR
3. **Follow code style** (see Code Standards below)
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Ensure tests pass** before submitting

### Pull Request Process

1. **Fork** the repository
2. **Create branch** for your changes
3. **Implement** your changes with tests
4. **Run full test suite**:
   ```bash
   python run_tests.py
   pytest tests/ --cov=spiraldelta
   ```
5. **Submit PR** with clear description
6. **Address review feedback**
7. **Squash commits** if requested

## 📋 Code Standards

### Python Code Style

We follow PEP 8 with some modifications:

```bash
# Format code
black src/ tests/ examples/
isort src/ tests/ examples/

# Check linting
flake8 src/ tests/ examples/
mypy src/
```

### Code Quality Guidelines

- **Docstrings**: All public functions must have docstrings
- **Type hints**: Use type hints for function signatures
- **Error handling**: Proper exception handling and meaningful error messages
- **Performance**: Consider performance implications, especially for core algorithms
- **Memory**: Be mindful of memory usage in vector operations

### Testing Requirements

- **Unit tests**: For all new functions and classes
- **Integration tests**: For database operations
- **Performance tests**: For core algorithms
- **Documentation tests**: Docstring examples should work

Example test structure:
```python
def test_spiral_coordinator_transform():
    """Test spiral coordinate transformation."""
    coordinator = SpiralCoordinator(dimensions=128)
    vector = np.random.randn(128)
    
    result = coordinator.transform(vector)
    
    assert isinstance(result, SpiralCoordinate)
    assert result.theta >= 0
    assert result.radius >= 0
    assert np.allclose(result.vector, vector)
```

## 🏗️ Development Areas

### High-Priority Areas

1. **Core Algorithm Optimization**
   - Spiral coordinate transformations
   - Delta encoding efficiency
   - Search performance improvements

2. **Storage Engine**
   - Memory-mapped file improvements
   - Compression algorithm enhancements
   - Index optimization

3. **API and Usability**
   - Python SDK improvements
   - Documentation and examples
   - Error handling and debugging

### Good First Issues

- Documentation improvements
- Example applications
- Test coverage expansion
- Performance benchmarking
- Bug fixes in edge cases

### Advanced Contributions

- New compression algorithms
- Alternative distance metrics
- Distributed system support
- GPU acceleration
- Language bindings (Rust, Go, etc.)

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests** (`tests/test_*.py`)
   - Individual component testing
   - Fast execution (< 1s per test)
   - No external dependencies

2. **Integration Tests** (`tests/integration/`)
   - Full database workflow testing
   - Real dataset processing
   - Moderate execution time (< 30s)

3. **Performance Tests** (`tests/benchmark/`)
   - Performance regression testing
   - Memory usage validation
   - Longer execution time acceptable

### Running Tests

```bash
# All tests
pytest tests/ -v

# Fast tests only
pytest tests/ -m "not slow"

# Specific component
pytest tests/test_spiral_coordinator.py -v

# With coverage
pytest tests/ --cov=spiraldelta --cov-report=html
```

### Test Data

- Use reproducible random seeds
- Include edge cases (empty, large, malformed data)
- Test with realistic embedding dimensions (300, 768, 1536)
- Validate compression and search quality

## 📚 Documentation

### Documentation Types

1. **API Documentation** (`docs/API.md`)
   - Complete method signatures
   - Parameter descriptions
   - Usage examples
   - Return value specifications

2. **User Guides** (`docs/GETTING_STARTED.md`)
   - Step-by-step tutorials
   - Common use cases
   - Best practices
   - Troubleshooting

3. **Code Documentation**
   - Inline comments for complex algorithms
   - Docstrings for all public APIs
   - Architecture explanations

### Documentation Standards

- **Clear examples**: Working code snippets
- **Complete coverage**: All public APIs documented
- **Accurate**: Keep docs synchronized with code
- **Accessible**: Beginner-friendly explanations

## 🚦 Development Workflow

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `perf/description` - Performance improvements
- `test/description` - Test additions

### Commit Message Format

```
type(scope): brief description

Detailed explanation of changes if needed.

- List specific changes
- Reference issues: Fixes #123
- Breaking changes noted
```

Types: `feat`, `fix`, `docs`, `test`, `perf`, `refactor`, `style`

### Release Process

1. **Version bumping**: Follow semantic versioning
2. **Changelog updates**: Document all changes
3. **Tag creation**: `git tag v1.2.3`
4. **Release notes**: Comprehensive change description

## 🐛 Debugging and Profiling

### Debugging Tools

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Profile performance
import cProfile
cProfile.run('your_function()')

# Memory profiling
from memory_profiler import profile
@profile
def your_function():
    pass
```

### Performance Profiling

```bash
# Profile script execution
python -m cProfile -o profile.stats your_script.py

# Analyze results
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## 🤝 Community Guidelines

### Communication

- **Be respectful** and professional
- **Be constructive** in feedback
- **Be patient** with newcomers
- **Be clear** in communication

### Getting Help

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Code Reviews**: Technical feedback and suggestions

### Recognition

Contributors are recognized in:
- Repository contributors list
- Release notes acknowledgments
- Documentation credits

## 📊 Performance Expectations

### Code Performance

- **Search queries**: < 10ms for 10K vectors
- **Insertion**: > 1000 vectors/second
- **Compression**: 30-70% storage reduction
- **Memory usage**: Reasonable for dataset size

### Test Performance

- **Unit tests**: < 5 minutes total
- **Integration tests**: < 15 minutes total
- **CI/CD pipeline**: < 30 minutes total

## 🔒 Security Considerations

- **No secrets in code**: Use environment variables
- **Input validation**: Sanitize all inputs
- **Dependencies**: Keep dependencies updated
- **Disclosure**: Report security issues privately

## 📜 License

By contributing to SpiralDeltaDB, you agree that your contributions will be licensed under the AGPL-3.0 License.

## 🙏 Thank You

Your contributions make SpiralDeltaDB better for everyone. Whether you're fixing a typo, adding a feature, or improving performance, every contribution is valuable!

For questions about contributing, feel free to:
- Open a GitHub Discussion
- Contact the maintainers
- Join our community conversations

Happy coding! 🌀