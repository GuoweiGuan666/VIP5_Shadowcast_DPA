#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_fake_user_generator.py

针对 fake_user_generator.py 的单元和 CLI 测试：
- 功能测试 read_lines, get_max_user_id, generate_fake_lines
- CLI 端到端：验证行过滤逻辑 & 映射扩展
- Direct Boost Attack 分支测试
"""
import unittest
import tempfile
import shutil
import os
import pickle
import subprocess
import sys
from pathlib import Path

from attack.baselines.DirectBoost_Random_Popular_attack.fake_user_generator import (
    read_lines, get_max_user_id, generate_fake_lines
)

class TestFakeUserGeneratorFunctions(unittest.TestCase):
    def setUp(self):
        self.lines = [
            "1 10 20 30 40 50",
            "2 11 12 13 14 15 16",
            "3 21 22 23 24 25 26",
        ]

    def test_read_and_max_id(self):
        f = tempfile.NamedTemporaryFile('w+', delete=False)
        f.write("\n".join(self.lines))
        f.close()
        try:
            out = read_lines(f.name)
            self.assertEqual(out, self.lines)
            self.assertEqual(get_max_user_id(out), 3)
        finally:
            os.unlink(f.name)

    def test_generate_fake_basic(self):
        fake = generate_fake_lines(
            orig_lines=self.lines,
            max_user_id=3,
            target_item=99,
            fake_count=5,
            min_history=2
        )
        self.assertEqual(len(fake), 5)
        for i, line in enumerate(fake):
            parts = line.split()
            self.assertEqual(parts[0], str(4 + i))
            self.assertEqual(parts[-1], "99")
            self.assertGreaterEqual(len(parts) - 2, 2)

    def test_generate_fake_insufficient(self):
        with self.assertRaises(RuntimeError):
            generate_fake_lines(self.lines, 3, 99, 1, min_history=100)

class TestFakeUserGeneratorCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data = self.tmp / "ds"
        self.data.mkdir()
        self.orig = ["1 10 20 30 40 50", "2 11 12 13 14 15 16"]
        with open(self.data / "sequential_data.txt", "w") as f:
            f.write("\n".join(self.orig))
        # 准备原始映射
        with open(self.data / "user_id2name.pkl", "wb") as f:
            pickle.dump({"1": "u1", "2": "u2"}, f)
        self.script = (Path(__file__).resolve().parent.parent /
                       "attack/baselines/DirectBoost_Random_Popular_attack/fake_user_generator.py")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_cli_outputs(self):
        # random_injection 分支: 因过滤阈值，最终只保留原始数据
        attack = "atk"
        mr = "0.1"
        poison_dir = self.data / "poisoned"
        poison_dir.mkdir(exist_ok=True)
        out_seq = poison_dir / f"sequential_data_{attack}_mr{mr}.txt"
        cmd = [
            sys.executable, str(self.script),
            "--input",  str(self.data / "sequential_data.txt"),
            "--output", str(out_seq),
            "--target_item", "99",
            "--fake_count", "1",
            "--min_history", "1",
            "--attack-name", attack,
            "--mr", mr
        ]
        p = subprocess.run(cmd, cwd=str(self.tmp),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(p.returncode, 0, msg=p.stderr)

        lines = out_seq.read_text().splitlines()
        self.assertListEqual(lines, self.orig)

        mfile = poison_dir / f"user_id2name_{attack}_mr{mr}.pkl"
        self.assertTrue(mfile.exists(), f"{mfile} 不存在")
        mp = pickle.load(open(mfile, "rb"))
        self.assertEqual(mp["1"], "u1")
        self.assertEqual(mp["2"], "u2")
        self.assertIn("3", mp)
        self.assertTrue(mp["3"].startswith("synthetic_user_"))

    def test_cli_direct_boost(self):
        # Direct Boost Attack: 新伪用户仅含 target_item，应被保留
        attack = "direct_boost"
        mr = "0.2"
        poison_dir = self.data / "poisoned"
        poison_dir.mkdir(exist_ok=True)
        out_seq = poison_dir / f"sequential_data_{attack}_mr{mr}.txt"
        cmd = [
            sys.executable, str(self.script),
            "--input",  str(self.data / "sequential_data.txt"),
            "--output", str(out_seq),
            "--target_item", "42",
            "--fake_count", "2",
            "--attack-name", attack,
            "--mr", mr
        ]
        p = subprocess.run(cmd, cwd=str(self.tmp),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(p.returncode, 0, msg=p.stderr)

        lines = out_seq.read_text().splitlines()
        # 应包含原始2行 + 2条伪用户记录
        self.assertEqual(len(lines), 4)
        # 验证新行格式为 "<uid> 42"
        new_lines = lines[2:]
        expected_uids = ["3", "4"]
        for uid, nl in zip(expected_uids, new_lines):
            parts = nl.split()
            self.assertEqual(parts, [uid, "42"]);

        # mapping 中应包含 uid 3,4
        mfile = poison_dir / f"user_id2name_{attack}_mr{mr}.pkl"
        self.assertTrue(mfile.exists(), f"{mfile} 不存在")
        mp = pickle.load(open(mfile, "rb"))
        for uid in expected_uids:
            self.assertIn(uid, mp)
            self.assertTrue(mp[uid].startswith("synthetic_user_"))

if __name__ == "__main__":
    unittest.main()
