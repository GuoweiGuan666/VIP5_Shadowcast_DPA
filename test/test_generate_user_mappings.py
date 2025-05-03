#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_generate_user_mappings.py

generate_user_mappings.py 的 CLI 测试：
- 准备 exp_splits_<suffix>.pkl 和 sequential_data_<suffix>.txt
- 调用脚本，检查 user_id2idx_<suffix>.pkl & user_id2name_<suffix>.pkl
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
        self.tmp = Path(tempfile.mkdtemp())
        ds = self.tmp / "beauty"
        ds.mkdir()
        self.suffix = "_atk_mr0.3"
        splits = {"train": [{"reviewerID": "1", "reviewerName": "u1"}]}
        with open(ds / f"exp_splits{self.suffix}.pkl", "wb") as f:
            pickle.dump(splits, f)
        with open(ds / f"sequential_data{self.suffix}.txt", "w") as f:
            f.write("1 10 20\n2 30 40\n")
        self.script = (Path(__file__).resolve().parent.parent /
                       "attack/baselines/DirectBoost_Random_Popular_attack/generate_user_mappings.py")
        self.ds = ds

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_generate_user_mappings(self):
        cmd = [
            sys.executable, str(self.script),
            "--attack-name", "atk",
            "--mr", "0.3",
            "--exp-splits", str(self.ds / f"exp_splits{self.suffix}.pkl"),
            "--seq-file",  str(self.ds / f"sequential_data{self.suffix}.txt"),
            "--output-dir", str(self.ds)
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(p.returncode, 0, msg=p.stderr)
        poisoned = self.ds / "poisoned"
        idx = pickle.load(open(poisoned / f"user_id2idx{self.suffix}.pkl","rb"))
        name= pickle.load(open(poisoned / f"user_id2name{self.suffix}.pkl","rb"))

        self.assertEqual(set(idx.keys()), {"1","2"})
        self.assertEqual(name["1"], "u1")
        self.assertEqual(name["2"], "<placeholder>")

if __name__ == "__main__":
    unittest.main()
