# src/pypi_lens/database.py
import os
from pathlib import Path
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging

import duckdb
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
# Constants
DB_PATH = os.path.join("data", "pypi_lens.db")
MODEL_NAME = 'all-MiniLM-L6-v2'


class PackageDB:
    def __init__(self, db_path: str = DB_PATH, embedder_model: str = MODEL_NAME):
        # Get absolute path, considering we're already in src/pypi_lens
        if not os.path.isabs(db_path):
            project_root = os.getcwd()  # This will be src/pypi_lens
            db_path = os.path.join(project_root, db_path)

        # Ensure data directory exists
        data_dir = os.path.dirname(db_path)
        os.makedirs(data_dir, exist_ok=True)

        self.db_path = db_path
        logger.info(f"Using database: {self.db_path}")

        # Initialize model with GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for embeddings")

        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        self.model_name = embedder_model
        self.embedding_dim = len(self.model.encode("test"))  # Get actual dimension

        self._init_db()
        self._check_model_consistency()

    def _init_db(self):
        """Initialize database schema with metadata table."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with duckdb.connect(self.db_path) as conn:
            # Create metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key VARCHAR PRIMARY KEY,
                    value VARCHAR
                );
            """)

            # Create packages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS packages (
                    name VARCHAR PRIMARY KEY,
                    description TEXT,
                    tags JSON,
                    version VARCHAR,
                    homepage VARCHAR,
                    repository VARCHAR,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    downloads INTEGER,
                    last_updated TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_downloads ON packages(downloads);
                CREATE INDEX IF NOT EXISTS idx_name ON packages(name);
            """)

            # Store model information
            conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES 
                    ('model_name', ?),
                    ('embedding_dim', ?)
            """, (self.model_name, str(self.embedding_dim)))

    def _check_model_consistency(self):
        """Check if current model matches the one used in database."""
        with duckdb.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT key, value FROM metadata 
                WHERE key IN ('model_name', 'embedding_dim')
            """).fetchall()

            if not result:
                return  # New database

            stored_info = dict(result)
            stored_model = stored_info.get('model_name')
            stored_dim = int(stored_info.get('embedding_dim', 0))

            if stored_model != self.model_name or stored_dim != self.embedding_dim:
                logger.warning(
                        f"Model mismatch. Stored: {stored_model} ({stored_dim}d), "
                        f"Current: {self.model_name} ({self.embedding_dim}d)"
                )
                self._regenerate_embeddings()

    def _regenerate_embeddings(self):
        """Regenerate embeddings with current model."""
        logger.info("Regenerating embeddings with current model...")

        with duckdb.connect(self.db_path) as conn:
            packages = conn.execute("""
                SELECT name, description, tags
                FROM packages
            """).fetchall()

            total = len(packages)
            for i, pkg in enumerate(packages, 1):
                name, description, tags = pkg
                tags_list = json.loads(tags)

                text_to_embed = f"{name} {description} {' '.join(tags_list)}"
                embedding = self.model.encode(text_to_embed)

                conn.execute("""
                    UPDATE packages 
                    SET embedding = ?,
                        embedding_dim = ?
                    WHERE name = ?
                """, (embedding.tobytes(), self.embedding_dim, name))

                if i % 100 == 0:
                    logger.info(f"Regenerated {i}/{total} embeddings...")

            # Update metadata
            conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES 
                    ('model_name', ?),
                    ('embedding_dim', ?)
            """, (self.model_name, str(self.embedding_dim)))

        logger.info("Embeddings regeneration complete")

    def fetch_package_info(self, package_name: str) -> Optional[Dict]:
        """Fetch package information from PyPI."""
        try:
            response = requests.get(
                    f"https://pypi.org/pypi/{package_name}/json",
                    timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Basic validation of response data
            info = data.get("info")
            if not info:
                return None

            name = info.get("name")
            description = info.get("description", "").strip()
            version = info.get("version")

            # Skip packages without basic info
            if not all([name, description, version]):
                return None

            # Extract tags from classifiers and keywords
            tags = []

            # Process classifiers
            classifiers = info.get("classifiers", [])
            if classifiers:
                for c in classifiers:
                    parts = c.split(" :: ")
                    if len(parts) > 1:
                        tags.append(parts[-1])

            # Process keywords
            keywords = info.get("keywords", "")
            if keywords:
                if isinstance(keywords, str):
                    tags.extend(k.strip() for k in keywords.split(",") if k.strip())
                elif isinstance(keywords, list):
                    tags.extend(k for k in keywords if k)

            # Get download count safely
            downloads = (
                    data.get("downloads", {}).get("last_month") or
                    data.get("downloads", {}).get("last_week") or
                    0
            )

            return {
                "name": name,
                "description": description,
                "tags": list(set(tag for tag in tags if tag)),  # Remove duplicates and empty tags
                "version": version,
                "homepage": info.get("home_page", ""),
                "repository": info.get("project_urls", {}).get("Source", ""),
                "downloads": downloads
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {package_name}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON for {package_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {package_name}: {str(e)}")
            return None

    def upsert_package(self, package_info: Dict):
        """Update or insert a package with embedding dimension."""
        try:
            text_to_embed = f"{package_info['name']} {package_info['description']} {' '.join(package_info['tags'])}"
            embedding = self.model.encode(text_to_embed)

            with duckdb.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO packages 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    package_info["name"],
                    package_info["description"],
                    json.dumps(package_info["tags"]),
                    package_info["version"],
                    package_info["homepage"],
                    package_info["repository"],
                    embedding.tobytes(),
                    self.embedding_dim,
                    package_info["downloads"],
                    datetime.now()
                ))

        except Exception as e:
            logger.error(f"Error upserting {package_info['name']}: {str(e)}")

    def search(self, query: str, limit: int = 25) -> List[Dict]:
        """Enhanced search that combines semantic similarity with keyword matching."""
        query_embedding = self.model.encode(query)
        query_terms = set(query.lower().split())

        results = []
        with duckdb.connect(self.db_path) as conn:
            # First check if we have any packages
            count = conn.execute("SELECT COUNT(*) FROM packages").fetchone()[0]
            if count == 0:
                logger.warning("Database is empty. Please run update_index.py first.")
                return []

            packages = conn.execute("""
                SELECT 
                    name, description, tags, version, 
                    homepage, repository, embedding, embedding_dim, downloads
                FROM packages 
                WHERE embedding IS NOT NULL
                AND embedding_dim = ?
            """, (self.embedding_dim,)).fetchall()

            for pkg in packages:
                name = pkg[0].lower()
                description = pkg[1].lower()
                tags = json.loads(pkg[2])
                tags_text = ' '.join(tags).lower()

                # Calculate semantic similarity
                pkg_embedding = np.frombuffer(pkg[6], dtype=np.float32)
                semantic_score = float(np.dot(query_embedding, pkg_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(pkg_embedding)
                ))

                # Calculate keyword matches
                keyword_score = 0
                for term in query_terms:
                    if term in name:
                        keyword_score += 0.4  # High weight for name matches
                    if term in tags_text:
                        keyword_score += 0.3  # Good weight for tag matches
                    if term in description:
                        keyword_score += 0.5  # High weight for description matches

                # Normalize keyword score
                keyword_score = min(keyword_score, 1.0)

                # Combined score (weighted average)
                final_score = (semantic_score * 0.6) + (keyword_score * 0.4)

                # Apply popularity boost (logarithmic scale)
                if pkg[8] > 0:  # if downloads > 0
                    popularity_boost = np.log10(pkg[8]) / 10
                    final_score *= (1 + popularity_boost)

                results.append({
                    "name": pkg[0],
                    "description": pkg[1][:200] + "..." if len(pkg[1]) > 200 else pkg[1],
                    "tags": tags[:5],
                    "version": pkg[3],
                    "homepage": pkg[4],
                    "repository": pkg[5],
                    "similarity": final_score,
                    "downloads": pkg[8]
                })

        # Sort by final score
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def get_stats(self) -> Dict:
        """Get database statistics."""
        with duckdb.connect(self.db_path) as conn:
            total_packages = conn.execute("SELECT COUNT(*) FROM packages").fetchone()[0]
            packages_with_embeddings = conn.execute(
                    "SELECT COUNT(*) FROM packages WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            last_updated = conn.execute(
                    "SELECT MAX(last_updated) FROM packages"
            ).fetchone()[0]

            return {
                "total_packages": total_packages,
                "packages_with_embeddings": packages_with_embeddings,
                "last_updated": last_updated
            }
