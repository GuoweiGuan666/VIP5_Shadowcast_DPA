#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_popular_mimicking_attack.py

新增 Popular Item Mimicking 攻击的单元测试：
- fake_user_generator.py 的 CLI 分支测试
- batch_poison.py 的集成测试
"""
import unittest
import tempfile
import shutil
import pickle
import subprocess
import sys
from pathlib import Path

class TestPopularMimickingFakeUserGeneratorCLI(unittest.TestCase):
    def setUp(self):
        # 创建临时目录和数据集
        self.tmp = Path(tempfile.mkdtemp())
        self.data = self.tmp / "ds"
        self.data.mkdir()
        # 原始序列文件（只包含 1 条简短序列）
        with open(self.data / "sequential_data.txt", "w") as f:
            f.write("1 10\n")
        # 原始映射
        with open(self.data / "user_id2name.pkl", "wb") as f:
            pickle.dump({"1": "u1"}, f)
        # 热门物品列表文件：包含三条记录
        self.pop_file = self.tmp / "high_pop.txt"
        with open(self.pop_file, "w") as f:
            f.write("高流行度候选目标物品列表：\n")
            f.write("    Item: AAA (ID: 100), Count: 50\n")
            f.write("    Item: BBB (ID: 200), Count: 40\n")
            f.write("    Item: CCC (ID: 300), Count: 30\n")
        # 脚本路径
        self.script = (
            Path(__file__).resolve().parent.parent /
            "attack/baselines/DirectBoost_Random_Popular_attack/fake_user_generator.py"
        )

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_cli_popular_mimicking(self):
        attack = "popular_mimicking"
        mr = "0.5"
        pop_k = "2"
        # 创建输出目录
        poison_dir = self.data / "poisoned"
        poison_dir.mkdir()
        out_seq = poison_dir / f"sequential_data_{attack}_mr{mr}.txt"
        # 调用 CLI
        cmd = [
            sys.executable, str(self.script),
            "--input", str(self.data / "sequential_data.txt"),
            "--output", str(out_seq),
            "--target_item", "999",
            "--fake_count", "2",
            "--attack-name", attack,
            "--mr", mr,
            "--pop-file", str(self.pop_file),
            "--pop-k", pop_k
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, cwd=str(self.tmp))
        self.assertEqual(p.returncode, 0, msg=p.stderr)

        # 检查合并后行数：原始 1 行 + fake_count = 2
        lines = out_seq.read_text().splitlines()
        self.assertEqual(len(lines), 3)
        # 原始行保持不变
        self.assertEqual(lines[0], "1 10")

        # 伪用户行格式：<new_uid> pop1 pop2 <target_item>
        for idx, fake in enumerate(lines[1:], start=1):
            parts = fake.split()
            # 第一列是新生成 user_id
            expected_uid = str(1 + idx)
            self.assertEqual(parts[0], expected_uid)
            # 中间是两个热门 id
            self.assertEqual(parts[1:-1], ["100", "200"])
            # 最后一列是 target_item
            self.assertEqual(parts[-1], "999")

    def test_mapping_extension(self):
        # 确保映射文件也正确写入
        attack = "popular_mimicking"
        mr = "0.5"
        pop_k = "1"
        poison_dir = self.data / "poisoned"
        poison_dir.mkdir(exist_ok=True)
        out_seq = poison_dir / f"sequential_data_{attack}_mr{mr}.txt"
        # 运行一次
        cmd = [
            sys.executable, str(self.script),
            "--input", str(self.data / "sequential_data.txt"),
            "--output", str(out_seq),
            "--target_item", "42",
            "--fake_count", "1",
            "--attack-name", attack,
            "--mr", mr,
            "--pop-file", str(self.pop_file),
            "--pop-k", pop_k
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, cwd=str(self.tmp))
        self.assertEqual(p.returncode, 0, msg=p.stderr)
        map_path = poison_dir / f"user_id2name_{attack}_mr{mr}.pkl"
        self.assertTrue(map_path.exists())
        mapping = pickle.load(open(map_path, "rb"))
        self.assertIn('1', mapping)
        self.assertIn('2', mapping)
        self.assertEqual(mapping['1'], 'u1')
        self.assertTrue(mapping['2'].startswith('synthetic_user_'))

class TestPopularMimickingBatchPoisonIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "data" / "beauty"
        self.root.mkdir(parents=True)
        # 原始交互序列：至少 6 tokens，满足默认 hist-min=5
        with open(self.root / "sequential_data.txt", "w") as f:
            f.write("1 10 20 30 40 50 60\n")
            f.write("2 11 12 13 14 15 16 17\n")
        # 准备热门列表
        self.pop_file = self.tmp / "high_pop_beauty.txt"
        with open(self.pop_file, "w") as f:
            f.write("高流行度候选目标物品列表：\n")
            f.write("    Item: X (ID: 7), Count: 100\n")
            f.write("    Item: Y (ID: 8), Count: 90\n")
        self.script = (
            Path(__file__).resolve().parent.parent /
            "attack/baselines/DirectBoost_Random_Popular_attack/batch_poison.py"
        )

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_batch_poison_popular_mimicking(self):
        # 使用 batch_poison 调用 popular_mimicking
        cmd = [
            sys.executable, str(self.script),
            "--attack-name", "popular_mimicking",
            "--mr", "0.5",
            "--pop-file", str(self.pop_file),
            "--pop-k", "1"
        ]
        p = subprocess.run(cmd, cwd=str(self.tmp),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(p.returncode, 0, msg=p.stderr)
        out = self.root / "poisoned" / "sequential_data_popular_mimicking_mr0.5.txt"
        self.assertTrue(out.exists())
        lines = out.read_text().splitlines()
        # 应包含原始 2 行 + fake_count = 1
        self.assertEqual(len(lines), 3)

        # 验证伪行格式：<new_uid> <pop_id> <target_item>
        fake = lines[2]
        parts = fake.split()
        # 第一列是新生成 user_id
        self.assertTrue(parts[0].isdigit())
        # 第二列是 pop_id
        self.assertEqual(parts[1], '7')
        # 最后一列是 target_item
        self.assertEqual(parts[-1], '2')

if __name__ == '__main__':
    unittest.main()
