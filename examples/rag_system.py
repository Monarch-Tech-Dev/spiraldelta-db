#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) System Example using SpiralDeltaDB.

This example demonstrates how to build a complete RAG system using SpiralDeltaDB
for efficient document storage and retrieval with semantic search capabilities.
"""

import numpy as np
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spiraldelta import SpiralDeltaDB


@dataclass
class Document:
    """Document structure for RAG system."""
    id: str
    title: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class SimpleEmbedder:
    """
    Simple embedding model simulation.
    
    In a real implementation, you would use models like:
    - sentence-transformers
    - OpenAI embeddings
    - Hugging Face transformers
    """
    
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        np.random.seed(42)  # For reproducible embeddings
        
        # Create a simple vocabulary-based embedding
        # In practice, use pre-trained models
        self.vocab = {}
        self.vocab_size = 10000
        
    def _text_to_features(self, text: str) -> np.ndarray:
        """Convert text to simple feature vector."""
        # Simple bag-of-words-like features
        words = re.findall(r'\w+', text.lower())
        
        # Create hash-based features for reproducibility
        features = np.zeros(self.dimensions)
        
        for word in words:
            # Use hash to map words to dimensions
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % self.dimensions
            features[idx] += 1.0
        
        # Add some semantic clustering based on text length and common words
        features[0] = len(words)  # Text length
        features[1] = text.count('the') + text.count('and') + text.count('is')
        features[2] = text.count('?')  # Questions
        features[3] = text.count('.')  # Statements
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        return self._text_to_features(text)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return np.array(embeddings)


class SpiralRAGSystem:
    """Complete RAG system using SpiralDeltaDB."""
    
    def __init__(
        self,
        db_path: str = "./rag_system.db",
        embedding_dimensions: int = 384,
        compression_ratio: float = 0.6,
    ):
        """Initialize RAG system."""
        self.embedder = SimpleEmbedder(dimensions=embedding_dimensions)
        
        # Initialize SpiralDeltaDB
        self.vector_db = SpiralDeltaDB(
            dimensions=embedding_dimensions,
            compression_ratio=compression_ratio,
            storage_path=db_path,
            auto_train_threshold=100,
            # Optimize for RAG workloads
            ef_construction=400,
            ef_search=100,
            distance_metric="cosine",
        )
        
        # Document storage
        self.documents = {}  # id -> Document
        self.doc_count = 0
        
        print(f"✓ RAG system initialized with {embedding_dimensions}D embeddings")
    
    def add_document(self, title: str, content: str, metadata: Optional[Dict] = None) -> str:
        """Add a document to the RAG system."""
        # Create document ID
        doc_id = f"doc_{self.doc_count}"
        self.doc_count += 1
        
        # Generate embedding
        full_text = f"{title} {content}"
        embedding = self.embedder.embed_text(full_text)
        
        # Create document
        doc_metadata = metadata or {}
        doc_metadata.update({
            "doc_id": doc_id,
            "title": title,
            "content_length": len(content),
            "word_count": len(content.split()),
        })
        
        document = Document(
            id=doc_id,
            title=title,
            content=content,
            metadata=doc_metadata,
            embedding=embedding
        )
        
        # Store document
        self.documents[doc_id] = document
        
        # Add to vector database
        vector_ids = self.vector_db.insert([embedding], [doc_metadata])
        document.metadata["vector_id"] = vector_ids[0]
        
        return doc_id
    
    def add_documents_batch(self, documents: List[Tuple[str, str, Dict]]) -> List[str]:
        """Add multiple documents efficiently."""
        print(f"Adding {len(documents)} documents to RAG system...")
        
        doc_ids = []
        embeddings = []
        metadatas = []
        
        for title, content, metadata in documents:
            # Create document ID
            doc_id = f"doc_{self.doc_count}"
            self.doc_count += 1
            
            # Generate embedding
            full_text = f"{title} {content}"
            embedding = self.embedder.embed_text(full_text)
            
            # Prepare metadata
            doc_metadata = metadata or {}
            doc_metadata.update({
                "doc_id": doc_id,
                "title": title,
                "content_length": len(content),
                "word_count": len(content.split()),
            })
            
            # Create document
            document = Document(
                id=doc_id,
                title=title,
                content=content,
                metadata=doc_metadata,
                embedding=embedding
            )
            
            self.documents[doc_id] = document
            doc_ids.append(doc_id)
            embeddings.append(embedding)
            metadatas.append(doc_metadata)
        
        # Batch insert into vector database
        vector_ids = self.vector_db.insert(np.array(embeddings), metadatas)
        
        # Update document metadata with vector IDs
        for doc_id, vector_id in zip(doc_ids, vector_ids):
            self.documents[doc_id].metadata["vector_id"] = vector_id
        
        print(f"✓ Added {len(doc_ids)} documents")
        return doc_ids
    
    def search_documents(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for relevant documents."""
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector database
        results = self.vector_db.search(
            query_embedding,
            k=k,
            filters=filters
        )
        
        # Retrieve full documents
        retrieved_docs = []
        for result in results:
            doc_id = result.metadata["doc_id"]
            document = self.documents[doc_id]
            
            doc_result = {
                "doc_id": doc_id,
                "title": document.title,
                "content": document.content,
                "similarity": result.similarity,
                "metadata": document.metadata.copy()
            }
            retrieved_docs.append(doc_result)
        
        return retrieved_docs
    
    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[Dict],
        max_context_length: int = 2000
    ) -> str:
        """
        Generate answer using retrieved context.
        
        In a real implementation, this would use an LLM like:
        - OpenAI GPT
        - Anthropic Claude
        - Local models via Ollama/llama.cpp
        """
        if not retrieved_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Build context from retrieved documents
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs):
            doc_text = f"Document {i+1}: {doc['title']}\n{doc['content'][:500]}..."
            
            if current_length + len(doc_text) > max_context_length:
                break
                
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        context = "\n\n".join(context_parts)
        
        # Simple rule-based answer generation (placeholder)
        # In practice, use an LLM here
        answer_parts = []
        
        # Extract key sentences from context
        sentences = []
        for doc in retrieved_docs[:3]:  # Use top 3 documents
            content_sentences = doc['content'].split('.')
            for sentence in content_sentences[:2]:  # First 2 sentences per doc
                if len(sentence.strip()) > 10:
                    sentences.append(sentence.strip())
        
        if sentences:
            answer = f"Based on the retrieved documents: {'. '.join(sentences[:3])}."
        else:
            answer = "The retrieved documents contain relevant information, but I cannot generate a specific answer with this simple implementation."
        
        return answer
    
    def ask(self, question: str, k: int = 5, filters: Optional[Dict] = None) -> Dict:
        """Complete RAG pipeline: retrieve and generate."""
        print(f"\n🤔 Question: {question}")
        
        # Retrieve relevant documents
        retrieved_docs = self.search_documents(question, k=k, filters=filters)
        
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": [],
                "retrieval_count": 0
            }
        
        print(f"📚 Retrieved {len(retrieved_docs)} relevant documents")
        
        # Generate answer
        answer = self.generate_answer(question, retrieved_docs)
        
        # Format response
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "title": doc["title"],
                "similarity": doc["similarity"],
                "content_preview": doc["content"][:200] + "...",
                "metadata": {k: v for k, v in doc["metadata"].items() 
                           if k not in ["content", "embedding"]}
            })
        
        response = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieval_count": len(retrieved_docs)
        }
        
        return response
    
    def get_statistics(self) -> Dict:
        """Get RAG system statistics."""
        db_stats = self.vector_db.get_stats()
        
        return {
            "total_documents": len(self.documents),
            "vector_count": db_stats.vector_count,
            "storage_size_mb": db_stats.storage_size_mb,
            "compression_ratio": db_stats.compression_ratio,
            "avg_query_time_ms": db_stats.avg_query_time_ms,
            "embedding_dimensions": self.embedder.dimensions,
        }
    
    def close(self):
        """Close the RAG system."""
        self.vector_db.close()


def create_sample_documents() -> List[Tuple[str, str, Dict]]:
    """Create sample documents for demonstration."""
    documents = [
        (
            "Introduction to Machine Learning",
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves. The primary goal is to allow computers to learn automatically without human intervention and adjust actions accordingly.",
            {"category": "AI/ML", "difficulty": "beginner", "topic": "overview"}
        ),
        (
            "Deep Learning Fundamentals",
            "Deep learning is a machine learning technique based on artificial neural networks with representation learning. It uses multiple layers to progressively extract higher-level features from raw input. Deep learning is particularly effective for image recognition, natural language processing, and speech recognition tasks.",
            {"category": "AI/ML", "difficulty": "intermediate", "topic": "deep_learning"}
        ),
        (
            "Vector Databases in AI",
            "Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for AI applications like semantic search, recommendation systems, and retrieval-augmented generation. Vector databases use similarity search algorithms to find relevant data points.",
            {"category": "Database", "difficulty": "intermediate", "topic": "vector_db"}
        ),
        (
            "Natural Language Processing",
            "Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on the interaction between computers and human language. NLP combines computational linguistics with machine learning and deep learning to help computers understand, interpret, and generate human language in a valuable way.",
            {"category": "AI/ML", "difficulty": "intermediate", "topic": "nlp"}
        ),
        (
            "Retrieval-Augmented Generation",
            "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language generation. It retrieves relevant documents from a knowledge base and uses them to inform the generation process, resulting in more accurate and factual responses. RAG is particularly useful for question-answering systems.",
            {"category": "AI/ML", "difficulty": "advanced", "topic": "rag"}
        ),
        (
            "Computer Vision Applications",
            "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects and react to what they see.",
            {"category": "AI/ML", "difficulty": "intermediate", "topic": "computer_vision"}
        ),
        (
            "Data Compression Techniques",
            "Data compression is the process of encoding information using fewer bits than the original representation. Compression can be either lossless or lossy. Lossless compression reduces bits by identifying and eliminating statistical redundancy, while lossy compression reduces bits by removing unnecessary or less important information.",
            {"category": "Computer Science", "difficulty": "intermediate", "topic": "compression"}
        ),
        (
            "Neural Network Architectures",
            "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Common architectures include feedforward networks, convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformers. Each architecture is suited for different types of tasks.",
            {"category": "AI/ML", "difficulty": "advanced", "topic": "neural_networks"}
        ),
        (
            "Semantic Search Systems",
            "Semantic search is a data searching technique that seeks to improve search accuracy by understanding the search intent and the contextual meaning of terms. Unlike traditional keyword-based search, semantic search considers the meaning behind words and can find relevant results even when exact keywords don't match.",
            {"category": "Search", "difficulty": "intermediate", "topic": "semantic_search"}
        ),
        (
            "Embeddings and Vector Representations",
            "In machine learning, embeddings are dense vector representations of discrete objects such as words, sentences, or images. These vectors capture semantic relationships and can be used for various tasks like similarity search, clustering, and classification. Word embeddings like Word2Vec and GloVe are popular examples.",
            {"category": "AI/ML", "difficulty": "intermediate", "topic": "embeddings"}
        ),
    ]
    
    return documents


def demo_rag_system():
    """Demonstrate the RAG system capabilities."""
    print("=" * 70)
    print("SpiralDeltaDB RAG System Demo")
    print("=" * 70)
    
    # Initialize RAG system
    print("\n1. Initializing RAG system...")
    rag = SpiralRAGSystem(
        db_path="./rag_demo.db",
        embedding_dimensions=384,
        compression_ratio=0.7
    )
    
    # Add sample documents
    print("\n2. Adding sample documents...")
    documents = create_sample_documents()
    doc_ids = rag.add_documents_batch(documents)
    print(f"✓ Added {len(doc_ids)} documents to knowledge base")
    
    # Show system statistics
    print("\n3. System Statistics...")
    stats = rag.get_statistics()
    print(f"✓ Documents: {stats['total_documents']}")
    print(f"✓ Storage: {stats['storage_size_mb']:.2f} MB")
    print(f"✓ Compression: {stats['compression_ratio']:.1%}")
    print(f"✓ Embeddings: {stats['embedding_dimensions']}D")
    
    # Demonstrate Q&A
    print("\n4. Question & Answer Demo...")
    
    questions = [
        "What is machine learning?",
        "How do vector databases work?",
        "What is retrieval-augmented generation?",
        "Explain deep learning",
        "What are neural networks?",
    ]
    
    for question in questions:
        response = rag.ask(question, k=3)
        
        print(f"\n💬 Answer: {response['answer']}")
        print(f"📖 Sources ({len(response['sources'])}):")
        
        for i, source in enumerate(response['sources'], 1):
            print(f"   {i}. {source['title']} (similarity: {source['similarity']:.3f})")
    
    # Demonstrate filtered search
    print("\n5. Filtered Search Demo...")
    
    # Search only in AI/ML category
    print("\n🔍 Searching only in AI/ML category...")
    ml_response = rag.ask(
        "What are the main AI techniques?",
        k=5,
        filters={"category": "AI/ML"}
    )
    
    print(f"💬 Answer: {ml_response['answer']}")
    print(f"📖 AI/ML Sources: {len(ml_response['sources'])}")
    
    # Search by difficulty level
    print("\n🔍 Searching intermediate level content...")
    intermediate_response = rag.ask(
        "Explain technical concepts",
        k=3,
        filters={"difficulty": "intermediate"}
    )
    
    print(f"📖 Intermediate Sources: {len(intermediate_response['sources'])}")
    
    # Performance demonstration
    print("\n6. Performance Test...")
    
    import time
    n_queries = 20
    query_times = []
    
    test_queries = [
        "machine learning algorithms",
        "neural network training",
        "vector similarity search",
        "data compression methods",
        "computer vision applications"
    ] * 4  # Repeat to get 20 queries
    
    for query in test_queries[:n_queries]:
        start_time = time.time()
        response = rag.ask(query, k=3)
        query_time = time.time() - start_time
        query_times.append(query_time)
    
    avg_time = np.mean(query_times) * 1000  # Convert to ms
    print(f"✓ Average query time: {avg_time:.2f}ms")
    print(f"✓ Query throughput: {1000/avg_time:.1f} queries/sec")
    
    # Final statistics
    print("\n7. Final Statistics...")
    final_stats = rag.get_statistics()
    print(f"✓ Total storage: {final_stats['storage_size_mb']:.2f} MB")
    print(f"✓ Compression achieved: {final_stats['compression_ratio']:.1%}")
    print(f"✓ Average retrieval time: {final_stats['avg_query_time_ms']:.2f}ms")
    
    # Cleanup
    rag.close()
    print("\n✅ RAG system demo completed!")


def advanced_rag_demo():
    """Demonstrate advanced RAG features."""
    print("\n" + "=" * 70)
    print("Advanced RAG Features Demo")
    print("=" * 70)
    
    # Initialize with advanced settings
    rag = SpiralRAGSystem(
        db_path="./advanced_rag.db",
        embedding_dimensions=512,  # Higher dimensions
        compression_ratio=0.8      # Higher compression
    )
    
    # Add more complex documents
    complex_documents = [
        (
            "Advanced Machine Learning Optimization",
            "Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent. Variants include stochastic gradient descent (SGD), Adam, and RMSprop. Learning rate scheduling and momentum can improve convergence. Regularization techniques like L1/L2 prevent overfitting.",
            {"category": "AI/ML", "difficulty": "advanced", "topic": "optimization", "keywords": ["gradient", "descent", "optimization"]}
        ),
        (
            "Transformer Architecture Deep Dive",
            "The Transformer architecture revolutionized NLP with its attention mechanism. Self-attention allows the model to weigh the importance of different words in a sequence. Multi-head attention captures different types of relationships. Position encoding handles sequence order. Layer normalization and residual connections stabilize training.",
            {"category": "AI/ML", "difficulty": "expert", "topic": "transformers", "keywords": ["attention", "transformer", "nlp"]}
        ),
        (
            "Vector Database Indexing Algorithms",
            "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search. It builds a multi-layer graph where higher layers have longer-range connections. Product quantization compresses vectors by splitting them into subspaces. IVF (Inverted File) uses clustering for efficient search.",
            {"category": "Database", "difficulty": "expert", "topic": "indexing", "keywords": ["hnsw", "quantization", "clustering"]}
        ),
    ]
    
    # Add complex documents
    rag.add_documents_batch(complex_documents)
    
    # Demonstrate complex queries
    complex_queries = [
        "How does attention mechanism work in transformers?",
        "What are the best optimization algorithms for neural networks?",
        "Explain HNSW algorithm for vector search",
    ]
    
    print("\nComplex Query Processing:")
    for query in complex_queries:
        response = rag.ask(query, k=2)
        print(f"\n🎯 Query: {query}")
        print(f"💡 Answer: {response['answer']}")
        
        # Show detailed source information
        for source in response['sources']:
            keywords = source['metadata'].get('keywords', [])
            print(f"   📄 {source['title']} | Keywords: {', '.join(keywords)}")
    
    rag.close()


def main():
    """Run the RAG system demonstration."""
    try:
        print("SpiralDeltaDB RAG System Examples")
        print("=" * 70)
        print("Demonstrating Retrieval-Augmented Generation with SpiralDeltaDB")
        print("Features:")
        print("• Efficient document storage and retrieval")
        print("• Semantic search with embeddings")
        print("• Compressed vector storage")
        print("• Metadata filtering")
        print("• High-performance querying")
        
        # Run demonstrations
        demo_rag_system()
        advanced_rag_demo()
        
        print("\n" + "=" * 70)
        print("✅ RAG System Examples Completed Successfully!")
        print("=" * 70)
        
        print("\nKey Features Demonstrated:")
        print("✓ Document ingestion and embedding")
        print("✓ Semantic similarity search")
        print("✓ Efficient storage with compression")
        print("✓ Metadata-based filtering")
        print("✓ Fast retrieval performance")
        print("✓ Integration with language generation")
        
        print("\nNext Steps:")
        print("• Integrate with real embedding models (sentence-transformers)")
        print("• Add support for LLM-based answer generation")
        print("• Implement document chunking for long texts")
        print("• Add support for multiple file formats")
        print("• Scale to production workloads")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ RAG demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())