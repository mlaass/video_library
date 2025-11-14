#!/usr/bin/env python3
"""
SQLite-based Face Database Management
Handles face identities, instances, and operations (merge/split)
"""

import sqlite3
import pickle
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np


class FaceDatabase:
    """Manages face recognition database with SQLite"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize database and create tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        # Face identities table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS face_identities (
                hash TEXT PRIMARY KEY,
                label TEXT,
                canonical_encoding BLOB NOT NULL,
                first_seen TIMESTAMP NOT NULL,
                last_seen TIMESTAMP NOT NULL,
                appearance_count INTEGER DEFAULT 0
            )
        """
        )

        # Face instances table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS face_instances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_hash TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                video_filename TEXT NOT NULL,
                timestamp REAL NOT NULL,
                detection_confidence REAL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (face_hash) REFERENCES face_identities(hash)
            )
        """
        )

        # Face operations table (merge/split history)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS face_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                target_hash TEXT,
                performed_at TIMESTAMP NOT NULL,
                notes TEXT
            )
        """
        )

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_instances_hash ON face_instances(face_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_instances_video ON face_instances(video_filename)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_identities_label ON face_identities(label)")

        self.conn.commit()

    def compute_face_hash(self, encoding: np.ndarray) -> str:
        """Compute hash from face encoding"""
        # Round to reduce floating point variations
        rounded = np.round(encoding, decimals=4)
        return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]

    def find_matching_identity(self, encoding: np.ndarray, tolerance: float = 0.6) -> Optional[str]:
        """Find existing face identity that matches the encoding within tolerance"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT hash, canonical_encoding FROM face_identities")

        for row in cursor.fetchall():
            stored_encoding = pickle.loads(row["canonical_encoding"])
            distance = np.linalg.norm(encoding - stored_encoding)
            if distance <= tolerance:
                return row["hash"]

        return None

    def add_identity(self, encoding: np.ndarray, label: Optional[str] = None) -> str:
        """Add new face identity to database"""
        face_hash = self.compute_face_hash(encoding)
        now = datetime.now().isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO face_identities 
            (hash, label, canonical_encoding, first_seen, last_seen, appearance_count)
            VALUES (?, ?, ?, ?, ?, 0)
        """,
            (face_hash, label, pickle.dumps(encoding), now, now),
        )

        self.conn.commit()
        return face_hash

    def add_instance(
        self, face_hash: str, image_filename: str, video_filename: str, timestamp: float, confidence: float = 1.0
    ) -> int:
        """Add face instance (detection) to database"""
        now = datetime.now().isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO face_instances 
            (face_hash, image_filename, video_filename, timestamp, detection_confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (face_hash, image_filename, video_filename, timestamp, confidence, now),
        )

        instance_id = cursor.lastrowid

        # Update identity stats
        cursor.execute(
            """
            UPDATE face_identities 
            SET appearance_count = appearance_count + 1,
                last_seen = ?
            WHERE hash = ?
        """,
            (now, face_hash),
        )

        self.conn.commit()
        return instance_id

    def update_label(self, face_hash: str, label: str) -> bool:
        """Update label for a face identity"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE face_identities 
            SET label = ?
            WHERE hash = ?
        """,
            (label, face_hash),
        )

        self.conn.commit()
        return cursor.rowcount > 0

    def get_identity(self, face_hash: str) -> Optional[Dict]:
        """Get face identity by hash"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM face_identities WHERE hash = ?", (face_hash,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def get_all_identities(self, labeled_only: bool = False, unlabeled_only: bool = False) -> List[Dict]:
        """Get all face identities with optional filtering"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM face_identities"
        if labeled_only:
            query += " WHERE label IS NOT NULL"
        elif unlabeled_only:
            query += " WHERE label IS NULL"
        query += " ORDER BY appearance_count DESC"

        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def get_instances_by_hash(self, face_hash: str) -> List[Dict]:
        """Get all instances for a face identity"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM face_instances 
            WHERE face_hash = ?
            ORDER BY created_at DESC
        """,
            (face_hash,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def get_instances_by_video(self, video_filename: str) -> List[Dict]:
        """Get all face instances for a video"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT fi.*, fid.label, fid.hash as face_hash
            FROM face_instances fi
            JOIN face_identities fid ON fi.face_hash = fid.hash
            WHERE fi.video_filename = ?
            ORDER BY fi.timestamp
        """,
            (video_filename,),
        )

        return [dict(row) for row in cursor.fetchall()]

    def merge_faces(self, source_hashes: List[str], target_hash: str, notes: str = "") -> bool:
        """Merge multiple face identities into one"""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        try:
            # Move all instances to target hash
            for source_hash in source_hashes:
                if source_hash == target_hash:
                    continue

                cursor.execute(
                    """
                    UPDATE face_instances 
                    SET face_hash = ?
                    WHERE face_hash = ?
                """,
                    (target_hash, source_hash),
                )

                # Record operation
                cursor.execute(
                    """
                    INSERT INTO face_operations 
                    (operation, source_hash, target_hash, performed_at, notes)
                    VALUES ('merge', ?, ?, ?, ?)
                """,
                    (source_hash, target_hash, now, notes),
                )

                # Delete old identity
                cursor.execute("DELETE FROM face_identities WHERE hash = ?", (source_hash,))

            # Update target identity stats
            cursor.execute(
                """
                UPDATE face_identities 
                SET appearance_count = (
                    SELECT COUNT(*) FROM face_instances WHERE face_hash = ?
                ),
                last_seen = ?
                WHERE hash = ?
            """,
                (target_hash, now, target_hash),
            )

            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"Error merging faces: {e}")
            return False

    def split_face(self, source_hash: str, instance_ids: List[int], new_label: Optional[str] = None) -> Optional[str]:
        """Split instances from one face identity into a new one"""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()

        try:
            # Get encoding from first instance to create new identity
            cursor.execute(
                """
                SELECT canonical_encoding FROM face_identities WHERE hash = ?
            """,
                (source_hash,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Create new identity with slightly modified encoding (to get new hash)
            encoding = pickle.loads(row["canonical_encoding"])
            # Add tiny random noise to create new hash
            new_encoding = encoding + np.random.normal(0, 0.0001, encoding.shape)
            new_hash = self.add_identity(new_encoding, new_label)

            # Move selected instances to new identity
            placeholders = ",".join("?" * len(instance_ids))
            cursor.execute(
                f"""
                UPDATE face_instances 
                SET face_hash = ?
                WHERE id IN ({placeholders})
            """,
                [new_hash] + instance_ids,
            )

            # Record operation
            cursor.execute(
                """
                INSERT INTO face_operations 
                (operation, source_hash, target_hash, performed_at, notes)
                VALUES ('split', ?, ?, ?, ?)
            """,
                (source_hash, new_hash, now, f"Split {len(instance_ids)} instances"),
            )

            # Update both identity stats
            for hash_val in [source_hash, new_hash]:
                cursor.execute(
                    """
                    UPDATE face_identities 
                    SET appearance_count = (
                        SELECT COUNT(*) FROM face_instances WHERE face_hash = ?
                    ),
                    last_seen = ?
                    WHERE hash = ?
                """,
                    (hash_val, now, hash_val),
                )

            self.conn.commit()
            return new_hash
        except Exception as e:
            self.conn.rollback()
            print(f"Error splitting face: {e}")
            return None

    def delete_identity(self, face_hash: str) -> bool:
        """Delete face identity and all its instances"""
        cursor = self.conn.cursor()

        try:
            cursor.execute("DELETE FROM face_instances WHERE face_hash = ?", (face_hash,))
            cursor.execute("DELETE FROM face_identities WHERE hash = ?", (face_hash,))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting identity: {e}")
            return False

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM face_identities")
        total_identities = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(*) as labeled FROM face_identities WHERE label IS NOT NULL")
        labeled = cursor.fetchone()["labeled"]

        cursor.execute("SELECT COUNT(*) as total FROM face_instances")
        total_instances = cursor.fetchone()["total"]

        cursor.execute("SELECT COUNT(DISTINCT video_filename) as videos FROM face_instances")
        videos_processed = cursor.fetchone()["videos"]

        return {
            "total_identities": total_identities,
            "labeled_identities": labeled,
            "unlabeled_identities": total_identities - labeled,
            "total_instances": total_instances,
            "videos_processed": videos_processed,
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
