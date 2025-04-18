#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
integration_data_loader_test.py
================================
该测试用例用于验证 VIP5_Dataset 类是否能够根据传入的参数加载正确的交互数据文件。

测试方法：
1. 构造一个模拟的命令行参数对象，其中包含 attack_mode、original_file、poisoned_file、data_root 等参数。
2. 分别设置 attack_mode 为 "none" 和非 "none"（如 "label"），构造对应的文件，并检查 VIP5_Dataset 读取的文件内容是否来自正确的文件。
"""

import os
import unittest
import tempfile

from collections import defaultdict
import numpy as np

# 假定 ReadLineFromFile、load_pickle、load_json、parse 已在 data.py 中定义
from src.data import VIP5_Dataset, ReadLineFromFile, load_pickle, load_json, parse

# 模拟配置一个简单的 all_tasks 和 task_list，测试中用不到具体细节
all_tasks = {'sequential': {}}
task_list = {'sequential': ['A-1']}
# 定义一个简单的 tokenizer 模拟
class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
    def encode(self, text, padding, truncation, max_length):
        return [1, 2, 3]
    def tokenize(self, text):
        return text.split()

tokenizer = DummyTokenizer()

class DummyArgs:
    pass

class TestVIP5DatasetDataLoading(unittest.TestCase):
    def setUp(self):
        # 创建临时文件夹来模拟数据目录结构
        self.temp_dir = tempfile.TemporaryDirectory()
        base_dir = self.temp_dir.name
        # 模拟 data/beauty 文件夹结构
        self.beauty_dir = os.path.join(base_dir, "beauty")
        os.makedirs(self.beauty_dir, exist_ok=True)
        # 创建原始数据文件
        self.original_file_path = os.path.join(self.beauty_dir, "sequential_data.txt")
        with open(self.original_file_path, "w", encoding="utf-8") as f:
            # 写入模拟内容，5 行数据
            f.write("1 10 20 30\n2 40 50 60\n3 70 80 90\n4 100 110 120\n5 130 140 150\n")
        # 创建投毒数据文件
        self.poisoned_file_path = os.path.join(self.beauty_dir, "sequential_data_poisoned.txt")
        with open(self.poisoned_file_path, "w", encoding="utf-8") as f:
            # 写入模拟内容，加上虚假行（这里简单重复原始数据再加上"999 2"虚假行做区分）
            original_lines = ReadLineFromFile(self.original_file_path)
            f.write("\n".join(original_lines) + "\n")
            # 假设投毒数据新增 2 行
            f.write("999 2\n")
            f.write("1000 2\n")
        # 创建 DummyArgs 对象，模拟从参数解析获得的参数
        self.args = DummyArgs()
        self.args.attack_mode = "none"
        self.args.original_file = "sequential_data.txt"
        self.args.poisoned_file = "sequential_data_poisoned.txt"
        self.args.data_root = base_dir
        self.args.image_feature_size_ratio = 2
        self.args.image_feature_type = "vitb32"
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    def test_load_original(self):
        # 当 attack_mode 为 "none" 时，应该加载原始数据文件
        self.args.attack_mode = "none"
        dataset = VIP5_Dataset(all_tasks, task_list, tokenizer, self.args, sample_numbers={}, mode='train', split="beauty", data_root=self.args.data_root)
        # 原始数据文件 5 行
        self.assertEqual(len(dataset.sequential_data), 5)
    
    def test_load_poisoned(self):
        # 当 attack_mode 不为 "none" 时，应该加载投毒数据文件
        self.args.attack_mode = "label"   # 任意非 "none" 值
        dataset = VIP5_Dataset(all_tasks, task_list, tokenizer, self.args, sample_numbers={}, mode='train', split="beauty", data_root=self.args.data_root)
        # 投毒数据文件应有 5 + 2 = 7 行数据
        self.assertEqual(len(dataset.sequential_data), 7)

if __name__ == '__main__':
    unittest.main()
