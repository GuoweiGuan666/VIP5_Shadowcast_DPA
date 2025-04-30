#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_fake_user_generator.py
============================
本单元测试文件用于测试 attack/baselines/direct_boost_attack/fake_user_generator.py 中各个函数的正确性，
包括：读取文件行、计算最大 user_id、生成虚假数据行、合并写入文件的功能，并测试添加历史长度(min_history)功能。
"""
import os
import tempfile
import unittest

from attack.baselines.direct_boost_attack.fake_user_generator import (
    read_lines, write_lines, get_max_user_id, generate_fake_lines
)

class TestFakeUserGenerator(unittest.TestCase):
    def setUp(self):
        # 创建包含多种长度交互序列的样本数据
        self.sample_data = (
            "1 1 2 3 4 5\n"
            "2 6 7 8 9 10 4 11\n"
            "3 4 12 13 14 15 16 17 18 19\n"
            "4 20 21 22 23 4 24\n"
            "5 4 25 26 27 28 29 30 31 32\n"
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "sample.txt")
        with open(self.input_file, "w", encoding="utf-8") as f:
            f.write(self.sample_data)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_lines(self):
        lines = read_lines(self.input_file)
        self.assertEqual(len(lines), 5)
        self.assertTrue(lines[0].startswith("1 "))

    def test_get_max_user_id(self):
        lines = read_lines(self.input_file)
        max_id = get_max_user_id(lines)
        # 样本中 user_id 最大为 5
        self.assertEqual(max_id, 5)

    def test_generate_fake_lines_basic(self):
        # 测试在最小历史长度为1时，生成格式正确的假数据
        lines = read_lines(self.input_file)
        max_id = get_max_user_id(lines)
        fake_lines = generate_fake_lines(
            orig_lines=lines,
            max_user_id=max_id,
            target_item="2",
            fake_count=3,
            min_history=1
        )
        self.assertEqual(len(fake_lines), 3)
        expected_ids = [str(max_id + i + 1) for i in range(3)]
        for i, line in enumerate(fake_lines):
            tokens = line.split()
            # 第一个是 user_id，最后一个是 target_item，中间至少一个历史
            self.assertEqual(tokens[0], expected_ids[i])
            self.assertEqual(tokens[-1], "2")
            self.assertGreaterEqual(len(tokens), 3)

    def test_generate_fake_lines_with_history(self):
        # 测试在指定较大历史长度时，历史数目正确且来自原交互序列
        lines = read_lines(self.input_file)
        max_id = get_max_user_id(lines)
        min_history = 3
        fake_count = 5
        target = "99"
        fake_lines = generate_fake_lines(
            orig_lines=lines,
            max_user_id=max_id,
            target_item=target,
            fake_count=fake_count,
            min_history=min_history
        )
        self.assertEqual(len(fake_lines), fake_count)
        # 收集所有原始历史 item
        all_items = []
        for l in lines:
            all_items.extend(l.split()[1:])
        for idx, line in enumerate(fake_lines):
            tokens = line.split()
            # 检查 user_id 和 target_item
            self.assertEqual(tokens[0], str(max_id + idx + 1))
            self.assertEqual(tokens[-1], target)
            # 检查历史长度
            history = tokens[1:-1]
            self.assertEqual(len(history), min_history)
            # 检查历史项来源于原始交互
            for h in history:
                self.assertIn(h, all_items)

    def test_generate_fake_lines_insufficient_history(self):
        # 当 min_history 大于所有原始序列长度时，应抛出 RuntimeError
        lines = read_lines(self.input_file)
        max_id = get_max_user_id(lines)
        with self.assertRaises(RuntimeError):
            generate_fake_lines(
                orig_lines=lines,
                max_user_id=max_id,
                target_item="5",
                fake_count=1,
                min_history=100
            )

if __name__ == '__main__':
    unittest.main()
