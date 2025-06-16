# Development Setup Guide

This guide shows how to properly set up both public and private repositories for Monarch AI development.

## 🚀 **Quick Setup**

### **Public Repository Only (Community Contributors)**

```bash
# Clone public repository
git clone https://github.com/monarch-ai/spiraldelta-db.git
cd spiraldelta-db

# Set up development environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
python run_tests.py
```

### **Both Repositories (Monarch AI Team)**

```bash
# Create separate development workspace
mkdir ~/monarch-development
cd ~/monarch-development

# Clone repositories in separate directories
git clone https://github.com/monarch-ai/spiraldelta-db.git
git clone git@github.com:monarch-ai/monarch-core.git  # Private SSH access

# Your workspace structure:
~/monarch-development/
├── spiraldelta-db/     # Public repository
└── monarch-core/       # Private repository (separate!)
```

## 🔧 **Development Environment Setup**

### **Public Repository Development**

```bash
cd ~/monarch-development/spiraldelta-db

# Create virtual environment for public development
python -m venv venv-public
source venv-public/bin/activate

# Install public dependencies
pip install -e ".[dev]"
pip install pytest black isort mypy flake8

# Set up git hooks
pre-commit install

# Run development tests
pytest tests/ -v
python examples/basic_usage.py
```

### **Private Repository Development**

```bash
cd ~/monarch-development/monarch-core

# Create separate virtual environment for private development
python -m venv venv-private
source venv-private/bin/activate

# Install private dependencies (including public spiraldelta-db)
pip install -e ../spiraldelta-db/  # Local development version
pip install -e ".[enterprise]"    # Private enterprise features

# Install additional enterprise dependencies
pip install -r requirements-enterprise.txt

# Set up enterprise configuration
cp .env.example .env
# Edit .env with your enterprise settings

# Run enterprise tests
pytest tests/enterprise/ -v
python examples/enterprise_demo.py
```

## 🛡️ **Security Best Practices**

### **Environment Isolation**

```bash
# Always use separate virtual environments
~/monarch-development/
├── spiraldelta-db/
│   └── venv-public/           # Public dependencies only
└── monarch-core/
    └── venv-private/          # Private + public dependencies
```

### **Git Configuration**

```bash
# In spiraldelta-db/ (public)
git config user.email "your-public@email.com"
git config user.name "Your Public Name"

# In monarch-core/ (private)  
git config user.email "your-enterprise@monarchai.com"
git config user.name "Your Name (Monarch AI)"
```

### **SSH Key Management**

```bash
# Set up separate SSH keys for different access levels
~/.ssh/
├── id_rsa_public      # For public GitHub access
├── id_rsa_monarch     # For private Monarch AI access
└── config

# SSH config (~/.ssh/config)
Host github.com-public
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa_public

Host github.com-monarch
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa_monarch

# Clone with specific keys
git clone git@github.com-public:monarch-ai/spiraldelta-db.git
git clone git@github.com-monarch:monarch-ai/monarch-core.git
```

## 🔄 **Development Workflow**

### **Working on Public Features**

```bash
cd ~/monarch-development/spiraldelta-db
source venv-public/bin/activate

# Create feature branch
git checkout -b feature/improve-spiral-algorithm

# Make changes to public algorithms
vim src/spiraldelta/spiral_coordinator.py

# Test public changes
pytest tests/test_spiral_coordinator.py -v
python examples/basic_usage.py

# Commit and push for community review
git add .
git commit -m "feat: improve spiral transformation by 20%"
git push origin feature/improve-spiral-algorithm

# Create pull request for community review
```

### **Working on Private Features**

```bash
cd ~/monarch-development/monarch-core
source venv-private/bin/activate

# Create feature branch
git checkout -b feature/advanced-ethics-layer

# Make changes to private features
vim soft_armor/ethics_validator.py

# Test integration with public spiraldelta-db
pytest tests/integration/test_ethics_integration.py -v

# Test enterprise features
python examples/enterprise_demo.py

# Commit and push to private repository
git add .
git commit -m "feat: add deepfake detection to Soft Armor"
git push origin feature/advanced-ethics-layer

# Direct merge (no public review needed)
```

### **Testing Integration**

```bash
# In monarch-core environment
cd ~/monarch-development/monarch-core
source venv-private/bin/activate

# Test that private features work with latest public code
pip install -e ../spiraldelta-db/  # Use local public development version

# Run integration tests
pytest tests/integration/ -v

# Test specific integration scenarios
python tests/integration/test_public_private_integration.py
```

## 📦 **Package Development**

### **Public Package Testing**

```bash
# Test public package installation
cd ~/monarch-development/spiraldelta-db
python setup.py sdist bdist_wheel

# Test installation in clean environment
python -m venv test-env
source test-env/bin/activate
pip install dist/spiraldelta_db-*.whl

# Verify public installation works
python -c "from spiraldelta import SpiralDeltaDB; print('Public package works!')"
```

### **Enterprise Package Testing**

```bash
# Test enterprise package
cd ~/monarch-development/monarch-core
python setup-enterprise.py sdist bdist_wheel

# Test in enterprise environment
python -m venv test-enterprise
source test-enterprise/bin/activate
pip install ../spiraldelta-db/dist/spiraldelta_db-*.whl  # Public dependency
pip install dist/monarch_core-*.whl                     # Enterprise package

# Verify enterprise features work
python -c "from monarch_core import MonarchVectorDB; print('Enterprise package works!')"
```

## 🚨 **Common Pitfalls to Avoid**

### ❌ **DON'T DO THIS**

```bash
# Never commit private code to public repository
cd spiraldelta-db/
mkdir private/  # ❌ WRONG
cp ../monarch-core/soft_armor/* private/  # ❌ DANGEROUS
git add private/  # ❌ SECURITY BREACH

# Never commit public repository inside private
cd monarch-core/
git submodule add https://github.com/monarch-ai/spiraldelta-db.git  # ❌ WRONG

# Never use same environment for both
pip install -e spiraldelta-db/ monarch-core/  # ❌ MIXING CONCERNS
```

### ✅ **DO THIS INSTEAD**

```bash
# Keep repositories completely separate
~/monarch-development/
├── spiraldelta-db/     # Public development
└── monarch-core/       # Private development (imports public via pip)

# Use proper dependency management
cd monarch-core/
pip install spiraldelta-db==1.0.0  # ✅ Published version
# OR
pip install -e ../spiraldelta-db/  # ✅ Local development version

# Clear separation of concerns
# Public: Core algorithms, community features
# Private: Enterprise features, ethics, conscious AI
```

## 📊 **Monitoring Development**

### **Public Repository Health**

```bash
# Check public repository status
cd spiraldelta-db/
git status                          # Clean working directory
pytest tests/ --cov=spiraldelta    # Test coverage
python scripts/benchmark.py        # Performance checks
```

### **Private Repository Health**

```bash
# Check enterprise integration
cd monarch-core/
git status                                    # Clean working directory  
pytest tests/integration/ -v                 # Integration tests
python scripts/enterprise_benchmark.py       # Enterprise performance
python scripts/security_audit.py             # Security validation
```

This setup ensures:
1. **Complete separation** between public and private code
2. **Secure development** practices
3. **Proper integration** testing
4. **Clean deployment** pipelines
5. **Team collaboration** without security risks