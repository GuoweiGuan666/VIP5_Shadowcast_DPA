#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test/test_generate_user_mappings.py

CLI-based unit test for attack/baselines/direct_boost_attack/generate_user_mappings.py.
Verifies that:
  - IDs only in sequential_data_poisoned.txt are included.
  - Names from exp_splits_poisoned.pkl are preserved.
  - New IDs get '<placeholder>' as name.
"""
import unittest
import tempfile
import shutil
import pickle
import subprocess
import sys
from pathlib import Path

class TestGenerateUserMappingsCLI(unittest.TestCase):
    def setUp(self):
        # 创建临时 data/<ds> 结构
        self.tmpdir = Path(tempfile.mkdtemp())
        self.ds_dir = self.tmpdir / "dummy"
        self.ds_dir.mkdir()

        # 1) 写 exp_splits_poisoned.pkl，只含 ID 'A'
        splits = {"train": [{"reviewerID": "A", "reviewerName": "Alice"}]}
        with open(self.ds_dir / "exp_splits_poisoned.pkl", "wb") as f:
            pickle.dump(splits, f)

        # 2) 写 sequential_data_poisoned.txt，含 'A' 和额外的 'B'
        lines = ["A 1 2 3", "B 4 5 6"]
        (self.ds_dir / "sequential_data_poisoned.txt").write_text("\n".join(lines))

    def tearDown(self):
        shutil.rmtree(str(self.tmpdir))

    def test_cli_generates_all_ids_with_placeholder(self):
        # 找到脚本
        project_root = Path(__file__).resolve().parent.parent
        script = project_root / "attack" / "baselines" / "direct_boost_attack" / "generate_user_mappings.py"

        # 调用 CLI
        proc = subprocess.run(
            [sys.executable, str(script), "--data-dir", str(self.ds_dir)],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.assertEqual(proc.returncode, 0, msg=f"STDERR:\n{proc.stderr}")

        # 检查输出文件
        idx_path  = self.ds_dir / "user_id2idx_poisoned.pkl"
        name_path = self.ds_dir / "user_id2name_poisoned.pkl"
        self.assertTrue(idx_path.exists(),  f"{idx_path} not found")
        self.assertTrue(name_path.exists(), f"{name_path} not found")

        user2idx  = pickle.load(open(idx_path,  "rb"))
        user2name = pickle.load(open(name_path, "rb"))

        # splits 里有 A，sequence 里多了 B
        self.assertEqual(set(user2idx.keys()), {"A", "B"})
        # 原名保留，新增 B 得到 placeholder
        self.assertEqual(user2name["A"], "Alice")
        self.assertEqual(user2name["B"], "<placeholder>")

if __name__ == "__main__":
    unittest.main()
