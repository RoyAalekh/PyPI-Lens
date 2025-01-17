# src/pypi_lens/update_index.py
import json
import os
from pathlib import Path
import requests
import logging
import time
from typing import List, Dict, Set
from database import PackageDB

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PyPIUpdater:
    PRIORITY_CATEGORIES = {
        'data-science': ['pandas', 'numpy', 'scipy', 'scikit-learn', 'matplotlib'],
        'web': ['django', 'flask', 'fastapi', 'requests', 'aiohttp'],
        'ml': ['tensorflow', 'torch', 'keras', 'transformers', 'xgboost'],
        'data': ['dask', 'vaex', 'polars', 'pyarrow', 'sqlalchemy'],
        'tools': ['pytest', 'black', 'mypy', 'poetry', 'pip'],
    }

    def __init__(self):
        self.db = PackageDB()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PyPI-Lens/1.0',
            'Accept': 'application/json'
        })

        # Create state directory if it doesn't exist
        self.state_dir = Path("state")
        self.state_dir.mkdir(exist_ok=True)
        self.state_file = self.state_dir / "update_state.json"
        self.packages_file = self.state_dir / "packages_list.json"

    def load_state(self) -> Dict:
        """Load the previous update state."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            "last_processed_index": 0,
            "success_count": 0,
            "error_count": 0,
            "last_update": None
        }

    def save_state(self, state: Dict):
        """Save the current update state."""
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def save_packages_list(self, packages: List[Dict]):
        """Save the full packages list."""
        with open(self.packages_file, 'w') as f:
            json.dump(packages, f)

    def load_packages_list(self) -> List[Dict]:
        """Load the saved packages list."""
        if self.packages_file.exists():
            with open(self.packages_file) as f:
                return json.load(f)
        return []

    def get_top_packages(self, limit: int = 5000) -> List[Dict]:
        """Get top packages by downloads."""
        try:
            response = self.session.get(
                    "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json",
                    timeout=30
            )
            response.raise_for_status()
            packages = response.json()["rows"][:limit]
            logger.info(f"Found {len(packages)} top packages")
            return packages
        except Exception as e:
            logger.error(f"Error fetching top packages: {e}")
            return []

    def get_priority_packages(self) -> Set[str]:
        """Get priority packages based on categories."""
        priority_packages = set()
        for category, pkgs in self.PRIORITY_CATEGORIES.items():
            priority_packages.update(pkgs)
        return priority_packages

    def update_index(self, max_packages: int = 10000, resume: bool = True):
        """Update package index with resume capability."""
        logger.info("Starting package index update")

        # Load previous state if resuming
        state = self.load_state() if resume else {
            "last_processed_index": 0,
            "success_count": 0,
            "error_count": 0,
            "last_update": None
        }

        # Load or fetch packages list
        all_packages = []
        if resume and self.packages_file.exists():
            all_packages = self.load_packages_list()
            logger.info(f"Resumed with {len(all_packages)} packages from saved state")
        else:
            # Get fresh package list
            top_packages = self.get_top_packages()
            priority_packages = self.get_priority_packages()

            # Combine packages
            all_packages = list(top_packages)
            top_names = {p["project"] for p in top_packages}

            for pkg_name in priority_packages:
                if pkg_name not in top_names:
                    all_packages.append({"project": pkg_name})

            # Save packages list for resume
            self.save_packages_list(all_packages)

        logger.info(f"""
Starting update:
- Total packages to process: {len(all_packages)}
- Starting from index: {state['last_processed_index']}
- Previous success count: {state['success_count']}
- Previous error count: {state['error_count']}
""")

        start_time = time.time()
        try:
            for i in range(state["last_processed_index"], len(all_packages)):
                if state["success_count"] >= max_packages:
                    logger.info(f"Reached maximum package limit: {max_packages}")
                    break

                # Rate limiting
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    if elapsed < 1:
                        time.sleep(1 - elapsed)
                    start_time = time.time()

                try:
                    pkg = all_packages[i]
                    pkg_name = pkg["project"]
                    pkg_info = self.db.fetch_package_info(pkg_name)

                    if pkg_info:
                        self.db.upsert_package(pkg_info)
                        state["success_count"] += 1

                        if state["success_count"] % 100 == 0:
                            logger.info(f"Processed {state['success_count']}/{max_packages} packages")

                except Exception as e:
                    state["error_count"] += 1
                    logger.debug(f"Error processing {pkg_name}: {str(e)}")
                    continue

                # Update state after each package
                state["last_processed_index"] = i + 1
                state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.save_state(state)

        except KeyboardInterrupt:
            logger.info("Update interrupted by user")
            self.save_state(state)
            return

        # Final statistics
        stats = self.db.get_stats()
        logger.info(f"""
Update completed:
- Total packages in database: {stats['total_packages']}
- Successful updates this run: {state['success_count']}
- Failed updates this run: {state['error_count']}
- Success rate: {(state['success_count']/(state['success_count']+state['error_count']))*100:.1f}%
""")

def main():
    updater = PyPIUpdater()
    updater.update_index(resume=True)  # Enable resume by default

if __name__ == "__main__":
    main()
