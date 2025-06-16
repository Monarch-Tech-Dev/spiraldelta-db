# SpiralDeltaDB Repository Architecture

## 🏗️ **Dual Repository Strategy**

SpiralDeltaDB follows a **dual repository architecture** that separates open-source foundations from proprietary enterprise features.

### **Public Repository: `spiraldelta-db`**
- **Visibility**: Public on GitHub
- **License**: AGPL-3.0 (Open Source)
- **Purpose**: Core vector database algorithms and community features
- **Access**: Open to all developers

### **Private Repository: `monarch-core`** 
- **Visibility**: Private (Monarch AI team only)
- **License**: Proprietary
- **Purpose**: Enterprise features, Soft Armor, Conscious Engine integrations
- **Access**: Restricted to Monarch AI team members

## 📦 **Package Distribution**

### **Public Package (PyPI)**
```bash
pip install spiraldelta-db
```
- Core SpiralDeltaDB functionality
- Community-driven features
- Open source algorithms
- Basic examples and documentation

### **Enterprise Package (Private Index)**
```bash
pip install monarch-core --extra-index-url https://private.monarchai.com/pypi
```
- Enterprise-grade features
- Soft Armor integration
- Conscious Engine connectivity
- Advanced monitoring and analytics
- Premium support

## 🔄 **Integration Model**

The private repository **imports and extends** the public package:

```python
# In monarch-core
from spiraldelta import SpiralDeltaDB  # Public foundation
from .soft_armor import EthicsLayer    # Private enhancement

class MonarchVectorDB(SpiralDeltaDB):
    """Enterprise vector database with ethics layer."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ethics = EthicsLayer()
        
    def insert(self, vectors, metadata=None):
        # Ethics checking before insertion
        self.ethics.validate_content(vectors, metadata)
        return super().insert(vectors, metadata)
```

## 🛡️ **Security Principles**

### **Separation Guarantees**
- ✅ Public repo has **zero knowledge** of private repo
- ✅ No private code accidentally committed to public
- ✅ No secrets or credentials in public repository
- ✅ Clear license boundaries (AGPL vs Proprietary)

### **Development Isolation**
- 🔒 Separate development environments
- 🔒 Different SSH keys and access controls
- 🔒 Independent CI/CD pipelines
- 🔒 Isolated dependency management

## 🎯 **Business Model**

### **Open Source Strategy**
- Build developer community around SpiralDeltaDB
- Establish technology leadership in vector databases
- Attract talent and contributions
- Create ecosystem around spiral ordering algorithms

### **Enterprise Value**
- Proprietary ethics and safety features
- Advanced enterprise integrations
- Premium support and consulting
- Conscious AI training capabilities

## 📈 **Community vs Enterprise**

| Feature | Public (spiraldelta-db) | Private (monarch-core) |
|---------|------------------------|------------------------|
| Core Algorithms | ✅ Open Source | ✅ Enhanced |
| Basic Search | ✅ Included | ✅ Optimized |
| Compression | ✅ Standard | ✅ Advanced |
| Ethics Layer | ❌ Not Included | ✅ Soft Armor |
| Conscious AI | ❌ Not Included | ✅ Full Integration |
| Enterprise Support | ❌ Community Only | ✅ Premium |
| Advanced Analytics | ❌ Basic Only | ✅ Full Suite |

This architecture allows us to:
1. **Build community** around solid open-source foundation
2. **Protect competitive advantages** in private repository
3. **Scale business model** with clear value differentiation
4. **Maintain security** through proper separation of concerns

The key principle: **Public repo is complete and valuable on its own**, while private repo adds enterprise-specific value on top.