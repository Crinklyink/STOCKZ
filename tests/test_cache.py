from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from stock_predictor.data.cache import SQLiteCache


class SQLiteCacheTests(unittest.TestCase):
    def test_concurrent_get_set_uses_same_db_without_open_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SQLiteCache(Path(tmpdir) / "cache" / "test_cache.db")

            def worker(index: int) -> int:
                key = f"key:{index}"
                cache.set(key, {"value": index}, ttl_seconds=60)
                payload = cache.get(key)
                return int(payload["value"])

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(worker, range(32)))

        self.assertEqual(results, list(range(32)))


if __name__ == "__main__":
    unittest.main()
