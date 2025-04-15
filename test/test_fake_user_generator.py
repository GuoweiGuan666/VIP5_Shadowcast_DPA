#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_fake_user_generator.py
============================
本单元测试文件用于测试 attack/baselines/direct_boost_attack/fake_user_generator.py 中各个函数的正确性，
包括：读取文件行、计算最大 user_id、生成虚假数据行和合并写入文件的功能。

为了测试，临时创建一个样本文件，内容如下（共 5 行）：
1 1 2 3 4 5
2 6 7 8 9 10 4 11
3 4 12 13 14 15 16 17 18 19
4 20 21 22 23 4 24
5 4 25 26 27 28 29 30 31 32
"""

import os
import tempfile
import unittest

# 更新 import 路径，导入新位置下的模块
from attack.baselines.direct_boost_attack.fake_user_generator import read_lines, write_lines, get_max_user_id, generate_fake_lines

class TestFakeUserGenerator(unittest.TestCase):
    def setUp(self):
        # 定义一个样本数据内容（每行以空白字符分隔）
        self.sample_data = (
            "1 1 2 3 4 5\n"
            "2 6 7 8 9 10 4 11\n"
            "3 4 12 13 14 15 16 17 18 19\n"
            "4 20 21 22 23 4 24\n"
            "5 4 25 26 27 28 29 30 31 32\n"
        )
        # 使用临时目录存放测试用的文件
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "sample.txt")
        with open(self.input_file, "w", encoding="utf-8") as f:
            f.write(self.sample_data)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_read_lines(self):
        lines = read_lines(self.input_file)
        self.assertEqual(len(lines), 5)
        self.assertTrue(lines[0].strip().startswith("1"))

    def test_get_max_user_id(self):
        lines = read_lines(self.input_file)
        max_id = get_max_user_id(lines)
        # 样本数据中的 user_id 分别为 1,2,3,4,5，最大 user_id 应为 5。
        self.assertEqual(max_id, 5)

    def test_generate_fake_lines(self):
        # 使用最大 user_id 为 5，生成 3 条虚假数据，目标物品设为 "2"
        fake_lines = generate_fake_lines(5, target_item="2", fake_count=3)
        self.assertEqual(len(fake_lines), 3)
        # 检查生成的虚假数据行格式，每行应为 "user_id target_item"
        expected_ids = ["6", "7", "8"]
        for i, line in enumerate(fake_lines):
            tokens = line.strip().split()
            self.assertEqual(tokens[0], expected_ids[i])
            self.assertEqual(tokens[1], "2")

    def test_integration_merge_and_write(self):
        # 读取原始数据行
        original_lines = read_lines(self.input_file)
        # 生成 3 条虚假数据
        fake_lines = generate_fake_lines(5, target_item="2", fake_count=3)
        # 合并两部分数据
        merged_lines = original_lines + fake_lines
        # 写入临时输出文件
        output_file = os.path.join(self.temp_dir.name, "merged.txt")
        write_lines(output_file, merged_lines)
        # 读取写入内容
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
        # 检查原始数据和虚假数据均在输出文件中
        for line in original_lines:
            self.assertIn(line.strip(), content)
        for line in fake_lines:
            self.assertIn(line.strip(), content)

if __name__ == '__main__':
    unittest.main()
