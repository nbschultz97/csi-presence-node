"""Tests for csi_node.data_collector — merge_datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from csi_node.data_collector import merge_datasets


class TestMergeDatasets:
    def test_merge_two_files(self, tmp_path):
        f1 = tmp_path / "a.npz"
        f2 = tmp_path / "b.npz"
        out = tmp_path / "merged.npz"

        np.savez(f1, X=np.ones((3, 5)), y=np.array([0, 0, 1]))
        np.savez(f2, X=np.zeros((2, 5)), y=np.array([2, 2]))

        merge_datasets([f1, f2], out)

        data = np.load(out)
        assert data["X"].shape == (5, 5)
        assert list(data["y"]) == [0, 0, 1, 2, 2]

    def test_skips_missing_file(self, tmp_path, capsys):
        f1 = tmp_path / "exists.npz"
        np.savez(f1, X=np.ones((2, 3)), y=np.array([0, 1]))
        out = tmp_path / "out.npz"

        merge_datasets([f1, tmp_path / "nope.npz"], out)

        data = np.load(out)
        assert data["X"].shape == (2, 3)

    def test_all_missing_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            merge_datasets([tmp_path / "a.npz", tmp_path / "b.npz"], tmp_path / "out.npz")
