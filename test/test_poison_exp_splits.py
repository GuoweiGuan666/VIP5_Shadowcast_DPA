#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_poison_exp_splits.py

poison_exp_splits.py 的 CLI 测试：
- 在临时 data/beauty 文件夹中准备 user_id2name.pkl,
  user_id2name_poisoned.pkl, exp_splits.pkl
- 调用脚本，检查 exp_splits_<attack>_mr<mr>.pkl 内容
"""
import unittest
import tempfile
import shutil
import pickle
import subprocess
import sys
import importlib.util
from pathlib import Path

HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestPoisonExpSplitsCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = self.tmp / "data" / "beauty"
        self.data.mkdir(parents=True)
        with open(self.data / "user_id2name.pkl", "wb") as f:
            pickle.dump({"1": "u1"}, f)
        with open(self.data / "user_id2name_poisoned.pkl", "wb") as f:
            pickle.dump({"1": "u1", "2": "u2"}, f)
        with open(self.data / "exp_splits.pkl", "wb") as f:
            pickle.dump({"train": [], "val": []}, f)
        self.script = (Path(__file__).resolve().parent.parent /
                       "attack/baselines/DirectBoost_Random_Popular_attack/poison_exp_splits.py")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_poison_exp_splits_output(self):
        cmd = [
            sys.executable, str(self.script),
            "--data-root", str(self.tmp / "data"),
            "--datasets", "beauty",
            "--target-asins", "beauty:ASIN1",
            "--attack-name", "atk",
            "--mr", "0.2"
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(p.returncode, 0, msg=p.stderr)
        out = self.data / "poisoned" / "exp_splits_atk_mr0.2.pkl"
        self.assertTrue(out.exists())
        new = pickle.load(open(out, "rb"))
        self.assertIn("train", new)
        self.assertEqual(len(new["train"]), 1)
        e = new["train"][0]
        self.assertEqual(e["reviewerID"], "2")
        self.assertEqual(e["reviewerName"], "u2")
        for key in ("asin","summary","overall","helpful","feature","explanation","reviewText"):
            self.assertIn(key, e)

if __name__ == "__main__":
    unittest.main()
