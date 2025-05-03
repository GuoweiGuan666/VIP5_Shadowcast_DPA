#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_batch_poison.py

batch_poison.py 的集成测试：
- 原始序列需至少 6 个 token，以满足默认 min_history=5
- 调用 batch_poison.py
- 验证生成了正确后缀的文件，行数与内容正确
"""
import unittest
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path

class TestBatchPoisonIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = self.tmp / "data" / "beauty"
        self.data.mkdir(parents=True)
        # 把每行扩到 6 个 token（5 条历史）以满足 min_history=5
        self.orig = [
            "1 10 20 30 40 50",
            "2 11 12 13 14 15"
        ]
        with open(self.data / "sequential_data.txt", "w") as f:
            f.write("\n".join(self.orig))
        self.script = (Path(__file__).resolve().parent.parent /
                       "attack/baselines/DirectBoost_Random_Popular_attack/batch_poison.py")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_batch_poison_creates_file(self):
        cmd = [sys.executable, str(self.script),
               "--attack-name", "rand", "--mr", "0.5"]
        p = subprocess.run(cmd, cwd=str(self.tmp),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(p.returncode, 0, msg=p.stderr)

        out = self.data / "poisoned" / "sequential_data_rand_mr0.5.txt"
        self.assertTrue(out.exists(), f"{out} 不存在")

        lines = out.read_text().splitlines()
        fake_count = int(len(self.orig) * 0.5)
        # 新文件应包含原始 + fake_count
        self.assertEqual(len(lines), len(self.orig) + fake_count)
        # 原始行在前
        self.assertListEqual(lines[:2], self.orig)
        # fake 行以 target_item "2" 结尾
        for fake in lines[-fake_count:]:
            self.assertEqual(fake.split()[-1], "2")

if __name__ == '__main__':
    unittest.main()
