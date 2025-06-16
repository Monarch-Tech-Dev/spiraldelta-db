"""
StorageEngine: Efficient storage backend for compressed vector data.

This module implements the storage layer that handles persisting and retrieving
compressed vector sequences, metadata, and index structures.
"""

import os
import mmap
import struct
import json
import sqlite3
import threading
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import logging
from contextlib import contextmanager
from .types import CompressedSequence, SearchResult, DatabaseStats
from .delta_encoder import DeltaEncoder

logger = logging.getLogger(__name__)


class StorageEngine:
    """
    High-performance storage backend for SpiralDeltaDB.
    
    Uses a hybrid approach combining SQLite for metadata and memory-mapped
    files for efficient binary data storage.
    """
    
    def __init__(
        self,
        storage_path: str,
        cache_size_mb: int = 512,
        enable_mmap: bool = True,
        page_size: int = 4096,
        delta_encoder: Optional[DeltaEncoder] = None,
    ):
        """
        Initialize StorageEngine.
        
        Args:
            storage_path: Database file path
            cache_size_mb: Memory cache size in MB
            enable_mmap: Enable memory-mapped file access
            page_size: Storage page size in bytes
        """
        self.storage_path = Path(storage_path)
        self.cache_size_mb = cache_size_mb
        self.enable_mmap = enable_mmap
        self.page_size = page_size
        
        # Create storage directory
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.db_path = self.storage_path.with_suffix('.db')
        self.data_path = self.storage_path.with_suffix('.data')
        self.index_path = self.storage_path.with_suffix('.idx')
        
        # Initialize storage
        self._init_database()
        self._init_data_file()
        
        # Memory-mapped file handle
        self._mmap_file = None
        self._mmap_obj = None
        
        # Thread-local storage for connections
        self._local = threading.local()
        
        # Delta encoder for vector compression/decompression
        self.delta_encoder = delta_encoder
        
        # Statistics
        self.read_count = 0
        self.write_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Initialized StorageEngine at {self.storage_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for metadata."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sequences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_offset INTEGER NOT NULL,
                    data_size INTEGER NOT NULL,
                    compression_ratio REAL NOT NULL,
                    vector_count INTEGER NOT NULL,
                    dimensions INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sequence_id INTEGER NOT NULL,
                    vector_index INTEGER NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (sequence_id) REFERENCES sequences (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_sequence ON vectors(sequence_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sequences_created ON sequences(created_at)")
            
            # Create encoder state table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS encoder_state (
                    id INTEGER PRIMARY KEY,
                    state_data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def _init_data_file(self) -> None:
        """Initialize binary data file."""
        if not self.data_path.exists():
            # Create empty data file
            with open(self.data_path, 'wb') as f:
                # Write file header
                header = struct.pack('<4sII', b'SDDB', 1, self.page_size)  # Magic, version, page_size
                f.write(header)
        
        # Open memory-mapped file if enabled
        if self.enable_mmap:
            self._open_mmap()
    
    def _open_mmap(self) -> None:
        """Open memory-mapped file for data access."""
        try:
            self._mmap_file = open(self.data_path, 'r+b')
            self._mmap_obj = mmap.mmap(self._mmap_file.fileno(), 0)
            logger.debug("Memory-mapped data file opened")
        except Exception as e:
            logger.warning(f"Failed to open memory-mapped file: {e}")
            self.enable_mmap = False
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(str(self.db_path))
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def store_vector_batch(
        self,
        vectors: List[np.ndarray],
        metadata: Optional[List[Dict]] = None
    ) -> List[int]:
        """
        Store vectors by compressing them and then storing.
        
        Args:
            vectors: List of vectors to store
            metadata: Optional metadata for each vector
            
        Returns:
            List of assigned vector IDs
        """
        if not self.delta_encoder:
            raise ValueError("DeltaEncoder required for vector storage")
        
        # Compress the vectors
        compressed = self.delta_encoder.encode_sequence(vectors)
        
        # Store the compressed sequence
        return self.store_compressed_batch(compressed, metadata)
    
    def store_compressed_batch(
        self, 
        compressed: CompressedSequence, 
        metadata: Optional[List[Dict]] = None
    ) -> List[int]:
        """
        Store compressed vector sequence and return vector IDs.
        
        Args:
            compressed: Compressed sequence to store
            metadata: Optional metadata for each vector
            
        Returns:
            List of assigned vector IDs
        """
        # Serialize compressed data
        data_bytes = self._serialize_compressed_sequence(compressed)
        
        # Write to data file
        data_offset = self._append_data(data_bytes)
        
        # Store metadata in database
        with self._transaction() as conn:
            # Insert sequence record
            cursor = conn.execute("""
                INSERT INTO sequences 
                (data_offset, data_size, compression_ratio, vector_count, dimensions, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data_offset,
                len(data_bytes),
                compressed.compression_ratio,
                compressed.metadata.get("sequence_length", len(compressed.anchors)),
                compressed.metadata.get("dimensions", compressed.anchors[0].shape[0] if compressed.anchors else 0),
                json.dumps(compressed.metadata)
            ))
            
            sequence_id = cursor.lastrowid
            
            # Insert vector records
            vector_ids = []
            vector_count = compressed.metadata.get("sequence_length", len(compressed.anchors))
            
            for i in range(vector_count):
                vector_metadata = metadata[i] if metadata and i < len(metadata) else {}
                
                cursor = conn.execute("""
                    INSERT INTO vectors (sequence_id, vector_index, metadata)
                    VALUES (?, ?, ?)
                """, (sequence_id, i, json.dumps(vector_metadata)))
                
                vector_ids.append(cursor.lastrowid)
        
        self.write_count += 1
        logger.debug(f"Stored compressed sequence with {len(vector_ids)} vectors")
        
        return vector_ids
    
    def _serialize_compressed_sequence(self, compressed: CompressedSequence) -> bytes:
        """
        Serialize compressed sequence to bytes.
        
        Args:
            compressed: CompressedSequence to serialize
            
        Returns:
            Serialized bytes
        """
        # Header: number of anchors, number of levels
        header = struct.pack('<II', len(compressed.anchors), len(compressed.delta_codes))
        data_parts = [header]
        
        # Serialize anchors
        for anchor in compressed.anchors:
            anchor_bytes = anchor.astype(np.float32).tobytes()
            anchor_header = struct.pack('<II', anchor.shape[0], len(anchor_bytes))
            data_parts.extend([anchor_header, anchor_bytes])
        
        # Serialize delta codes
        for level_codes in compressed.delta_codes:
            level_header = struct.pack('<I', len(level_codes))
            data_parts.append(level_header)
            
            for codes in level_codes:
                codes_bytes = codes.astype(np.uint8).tobytes()
                codes_header = struct.pack('<II', len(codes), len(codes_bytes))
                data_parts.extend([codes_header, codes_bytes])
        
        # Serialize metadata
        metadata_json = json.dumps(compressed.metadata).encode('utf-8')
        metadata_header = struct.pack('<I', len(metadata_json))
        data_parts.extend([metadata_header, metadata_json])
        
        return b''.join(data_parts)
    
    def _append_data(self, data: bytes) -> int:
        """
        Append data to data file and return offset.
        
        Args:
            data: Data bytes to append
            
        Returns:
            Offset where data was written
        """
        with open(self.data_path, 'ab') as f:
            offset = f.tell()
            f.write(data)
            return offset
    
    def load_compressed_sequence(self, sequence_id: int) -> CompressedSequence:
        """
        Load compressed sequence by ID.
        
        Args:
            sequence_id: Sequence ID to load
            
        Returns:
            CompressedSequence object
        """
        conn = self._get_connection()
        
        # Get sequence metadata
        row = conn.execute(
            "SELECT * FROM sequences WHERE id = ?", 
            (sequence_id,)
        ).fetchone()
        
        if not row:
            raise ValueError(f"Sequence {sequence_id} not found")
        
        # Read data from file
        data_bytes = self._read_data(row['data_offset'], row['data_size'])
        
        # Deserialize compressed sequence
        compressed = self._deserialize_compressed_sequence(data_bytes)
        
        self.read_count += 1
        return compressed
    
    def _read_data(self, offset: int, size: int) -> bytes:
        """
        Read data from storage file.
        
        Args:
            offset: File offset
            size: Number of bytes to read
            
        Returns:
            Data bytes
        """
        if self.enable_mmap and self._mmap_obj:
            # Use memory-mapped access
            return self._mmap_obj[offset:offset + size]
        else:
            # Use regular file I/O
            with open(self.data_path, 'rb') as f:
                f.seek(offset)
                return f.read(size)
    
    def _deserialize_compressed_sequence(self, data: bytes) -> CompressedSequence:
        """
        Deserialize compressed sequence from bytes.
        
        Args:
            data: Serialized data bytes
            
        Returns:
            CompressedSequence object
        """
        offset = 0
        
        # Read header
        n_anchors, n_levels = struct.unpack('<II', data[offset:offset + 8])
        offset += 8
        
        # Read anchors
        anchors = []
        for _ in range(n_anchors):
            dims, byte_size = struct.unpack('<II', data[offset:offset + 8])
            offset += 8
            
            anchor_bytes = data[offset:offset + byte_size]
            anchor = np.frombuffer(anchor_bytes, dtype=np.float32).reshape(dims)
            anchors.append(anchor)
            offset += byte_size
        
        # Read delta codes
        delta_codes = []
        for _ in range(n_levels):
            n_codes = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            
            level_codes = []
            for _ in range(n_codes):
                code_len, byte_size = struct.unpack('<II', data[offset:offset + 8])
                offset += 8
                
                codes_bytes = data[offset:offset + byte_size]
                codes = np.frombuffer(codes_bytes, dtype=np.uint8)
                level_codes.append(codes)
                offset += byte_size
            
            delta_codes.append(level_codes)
        
        # Read metadata
        metadata_size = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        
        metadata_json = data[offset:offset + metadata_size].decode('utf-8')
        metadata = json.loads(metadata_json)
        
        return CompressedSequence(
            anchors=anchors,
            delta_codes=delta_codes,
            metadata=metadata,
            compression_ratio=metadata.get('compression_ratio', 0.0)
        )
    
    def get_vectors_by_ids(self, vector_ids: List[int]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve multiple vectors by IDs efficiently.
        
        Args:
            vector_ids: List of vector IDs to retrieve
            
        Returns:
            List of (vector, metadata) tuples
        """
        if not vector_ids:
            return []
        
        conn = self._get_connection()
        
        # Get all vector metadata in one query
        placeholders = ','.join(['?'] * len(vector_ids))
        rows = conn.execute(f"""
            SELECT v.*, s.* FROM vectors v
            JOIN sequences s ON v.sequence_id = s.id
            WHERE v.id IN ({placeholders})
            ORDER BY v.sequence_id, v.vector_index
        """, vector_ids).fetchall()
        
        if not rows:
            return []
        
        # Group by sequence to minimize decompression
        sequences_cache = {}
        results = []
        
        for row in rows:
            sequence_id = row['sequence_id']
            vector_index = row['vector_index']
            
            # Load sequence if not cached
            if sequence_id not in sequences_cache:
                compressed = self.load_compressed_sequence(sequence_id)
                if self.delta_encoder:
                    decoded_vectors = self.delta_encoder.decode_sequence(compressed)
                    sequences_cache[sequence_id] = decoded_vectors
                else:
                    logger.warning("No DeltaEncoder available for vector decoding")
                    sequences_cache[sequence_id] = [np.zeros(row['dimensions']) for _ in range(row['vector_count'])]
            
            # Get the specific vector
            decoded_vectors = sequences_cache[sequence_id]
            if 0 <= vector_index < len(decoded_vectors):
                vector = decoded_vectors[vector_index]
            else:
                vector = np.zeros(row['dimensions'])
            
            vector_metadata = json.loads(row['metadata']) if row['metadata'] else {}
            results.append((vector, vector_metadata))
        
        return results
    
    def get_vector_by_id(self, vector_id: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Retrieve specific vector by ID.
        
        Args:
            vector_id: Vector ID to retrieve
            
        Returns:
            Tuple of (vector, metadata)
        """
        conn = self._get_connection()
        
        # Get vector metadata
        row = conn.execute("""
            SELECT v.*, s.* FROM vectors v
            JOIN sequences s ON v.sequence_id = s.id
            WHERE v.id = ?
        """, (vector_id,)).fetchone()
        
        if not row:
            raise ValueError(f"Vector {vector_id} not found")
        
        # Load the full compressed sequence
        compressed = self.load_compressed_sequence(row['sequence_id'])
        
        # Decode the specific vector from compressed sequence
        vector_metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        if self.delta_encoder:
            # Decode the full sequence to get the specific vector
            decoded_vectors = self.delta_encoder.decode_sequence(compressed)
            vector_index = row['vector_index']
            
            if 0 <= vector_index < len(decoded_vectors):
                vector = decoded_vectors[vector_index]
            else:
                raise ValueError(f"Vector index {vector_index} out of range")
        else:
            # Fallback: return zero vector if no encoder available
            logger.warning("No DeltaEncoder available for vector decoding")
            vector = np.zeros(row['dimensions'])
        
        return vector, vector_metadata
    
    def get_all_vectors_in_sequence(self, sequence_id: int) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get all vectors in a compressed sequence efficiently.
        
        Args:
            sequence_id: Sequence ID
            
        Returns:
            List of (vector, metadata) tuples
        """
        conn = self._get_connection()
        
        # Get all vector metadata for this sequence
        rows = conn.execute("""
            SELECT * FROM vectors 
            WHERE sequence_id = ? 
            ORDER BY vector_index
        """, (sequence_id,)).fetchall()
        
        if not rows:
            return []
        
        # Load and decode the sequence once
        compressed = self.load_compressed_sequence(sequence_id)
        if self.delta_encoder:
            decoded_vectors = self.delta_encoder.decode_sequence(compressed)
        else:
            # Fallback to zero vectors
            dimensions = compressed.metadata.get('dimensions', 128)
            vector_count = compressed.metadata.get('sequence_length', len(rows))
            decoded_vectors = [np.zeros(dimensions) for _ in range(vector_count)]
        
        # Combine vectors with metadata
        results = []
        for row in rows:
            vector_index = row['vector_index']
            if 0 <= vector_index < len(decoded_vectors):
                vector = decoded_vectors[vector_index]
            else:
                vector = np.zeros(len(decoded_vectors[0]) if decoded_vectors else 128)
            
            vector_metadata = json.loads(row['metadata']) if row['metadata'] else {}
            results.append((vector, vector_metadata))
        
        return results
    
    def delete_sequence(self, sequence_id: int) -> bool:
        """
        Delete compressed sequence and associated vectors.
        
        Args:
            sequence_id: Sequence ID to delete
            
        Returns:
            True if deleted successfully
        """
        with self._transaction() as conn:
            # Delete vectors first (foreign key constraint)
            conn.execute("DELETE FROM vectors WHERE sequence_id = ?", (sequence_id,))
            
            # Delete sequence
            cursor = conn.execute("DELETE FROM sequences WHERE id = ?", (sequence_id,))
            
            return cursor.rowcount > 0
    
    def get_database_stats(self) -> DatabaseStats:
        """
        Get comprehensive database statistics.
        
        Returns:
            DatabaseStats object
        """
        conn = self._get_connection()
        
        # Get vector count
        vector_count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        
        # Get storage sizes
        storage_size_mb = self.data_path.stat().st_size / (1024 * 1024) if self.data_path.exists() else 0
        index_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        
        # Get average compression ratio
        avg_compression = conn.execute(
            "SELECT AVG(compression_ratio) FROM sequences"
        ).fetchone()[0] or 0.0
        
        # Get dimensions (from most recent sequence)
        dimensions_row = conn.execute(
            "SELECT dimensions FROM sequences ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        dimensions = dimensions_row[0] if dimensions_row else 0
        
        # Estimate memory usage
        memory_usage_mb = (
            storage_size_mb * 0.1 +  # Assume 10% of data in memory
            index_size_mb +           # Full index in memory
            self.cache_size_mb * 0.5  # Assume 50% cache utilization
        )
        
        return DatabaseStats(
            vector_count=vector_count,
            storage_size_mb=storage_size_mb,
            compression_ratio=avg_compression,
            avg_query_time_ms=0.0,  # Would be tracked by search engine
            index_size_mb=index_size_mb,
            memory_usage_mb=memory_usage_mb,
            dimensions=dimensions,
        )
    
    def compact_storage(self) -> None:
        """
        Compact storage by removing deleted data and optimizing layout.
        """
        logger.info("Starting storage compaction")
        
        # SQLite vacuum
        conn = self._get_connection()
        conn.execute("VACUUM")
        
        # Data file compaction would require rebuilding the file
        # This is a complex operation that would involve:
        # 1. Reading all valid sequences
        # 2. Writing them to a new file
        # 3. Updating offsets in database
        # 4. Replacing old file with new file
        
        logger.info("Storage compaction completed")
    
    def backup(self, backup_path: str) -> None:
        """
        Create backup of database.
        
        Args:
            backup_path: Path for backup files
        """
        backup_dir = Path(backup_path)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup SQLite database
        import shutil
        shutil.copy2(self.db_path, backup_dir / f"{self.db_path.name}.backup")
        shutil.copy2(self.data_path, backup_dir / f"{self.data_path.name}.backup")
        
        logger.info(f"Database backed up to {backup_path}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage engine statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        return {
            "storage_path": str(self.storage_path),
            "cache_size_mb": self.cache_size_mb,
            "enable_mmap": self.enable_mmap,
            "page_size": self.page_size,
            "read_count": self.read_count,
            "write_count": self.write_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
        }
    
    def close(self) -> None:
        """Close storage engine and cleanup resources."""
        if self._mmap_obj:
            self._mmap_obj.close()
        if self._mmap_file:
            self._mmap_file.close()
        
        # Close thread-local connections
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
        
        logger.info("StorageEngine closed")
    
    def save_encoder_state(self) -> None:
        """
        Save the current DeltaEncoder state to database.
        """
        if not self.delta_encoder:
            logger.warning("No DeltaEncoder to save")
            return
        
        state_data = self.delta_encoder.save_state()
        
        with self._transaction() as conn:
            # Remove old state
            conn.execute("DELETE FROM encoder_state")
            
            # Insert new state
            conn.execute(
                "INSERT INTO encoder_state (state_data) VALUES (?)",
                (state_data,)
            )
        
        logger.info("DeltaEncoder state saved")
    
    def load_encoder_state(self) -> Optional[DeltaEncoder]:
        """
        Load DeltaEncoder state from database.
        
        Returns:
            Loaded DeltaEncoder or None if no state found
        """
        conn = self._get_connection()
        
        row = conn.execute(
            "SELECT state_data FROM encoder_state ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        
        if not row:
            logger.info("No encoder state found in database")
            return None
        
        try:
            encoder = DeltaEncoder.load_state(row['state_data'])
            logger.info("DeltaEncoder state loaded successfully")
            return encoder
        except Exception as e:
            logger.error(f"Failed to load encoder state: {e}")
            return None
    
    def set_delta_encoder(self, encoder: DeltaEncoder) -> None:
        """
        Set the DeltaEncoder for this storage engine.
        
        Args:
            encoder: Trained DeltaEncoder instance
        """
        self.delta_encoder = encoder
        logger.info("DeltaEncoder set for storage engine")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()