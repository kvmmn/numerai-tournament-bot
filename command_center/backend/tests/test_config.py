from __future__ import annotations

import unittest
from pathlib import Path

from app.core.config import default_data_dir


class DataDirectoryResolutionTests(unittest.TestCase):
    def test_installed_runtime_defaults_to_sibling_data_directory(self):
        backend = Path(
            "/Users/example/Library/Application Support/"
            "Numerai/runtime/backend"
        )
        result = default_data_dir(backend, Path("/unused/root"))
        self.assertEqual(
            result,
            Path(
                "/Users/example/Library/Application Support/Numerai/data"
            ),
        )

    def test_source_checkout_keeps_existing_repository_root_default(self):
        backend = Path("/workspace/command_center/backend")
        result = default_data_dir(backend, Path("/workspace"))
        self.assertEqual(result, Path("/workspace"))


if __name__ == "__main__":
    unittest.main()
