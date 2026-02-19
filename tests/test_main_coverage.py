"""Tests for csi_node.__main__ module entry point."""
from __future__ import annotations

import unittest
from unittest.mock import patch


class TestMainModule(unittest.TestCase):
    @patch("run.main")
    def test_main_calls_run_main(self, mock_main):
        """__main__.py should call run.main()."""
        import importlib
        import csi_node.__main__
        # The module-level code runs on import, calling main()
        # We just verify the import doesn't crash
        mock_main.assert_called()


if __name__ == "__main__":
    unittest.main()
