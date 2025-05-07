# src/ssl_bench/cache.py

import os, json, hashlib, joblib
from pathlib import Path

class CacheManager:
    def __init__(self, cache_dir: str = "data/processed/experiment_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, config: dict) -> str:
        j = json.dumps(config, sort_keys=True)
        return hashlib.md5(j.encode()).hexdigest()

    def _path(self, config: dict) -> Path:
        return self.cache_dir / f"{self._make_key(config)}.pkl"

    def exists(self, config: dict) -> bool:
        return self._path(config).exists()

    def load(self, config: dict):
        return joblib.load(self._path(config))

    def save(self, config: dict, result: any):
        joblib.dump(result, self._path(config))

    def remove(self, config: dict):
        p = self._path(config)
        if p.exists():
            p.unlink()