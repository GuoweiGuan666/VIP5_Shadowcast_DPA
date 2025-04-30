#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：test_poison_exp_splits.py
===============================
对 poison_exp_splits.py 脚本进行单元测试。

测试会在临时目录中模拟一个小型数据集，并验证：
- 能正确识别原始与中毒用户映射
- 只在 train 拆分中追加伪用户条目
- 生成的 exp_splits_poisoned.pkl 存在且内容符合预期

使用方法：
```bash
cd /scratch/guanguowei/Code/MyWork/VIP5_Shadowcast_DPA
# 安装依赖：
# pip install pytest
# 运行测试：
python -m unittest discover -s test
# 或：pytest test
```

新增：
- 验证伪用户点评条目中含有 reviewerID、reviewerName 和 summary 字段
"""
import os
import sys
import tempfile
import shutil
import pickle
import unittest

# 确保直接加载 poison_exp_splits.py 脚本
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPT_DIR = os.path.join(PROJECT_ROOT, 'attack', 'baselines', 'direct_boost_attack')
sys.path.insert(0, SCRIPT_DIR)

from poison_exp_splits import main as poison_main

class TestPoisonExpSplits(unittest.TestCase):
    def setUp(self):
        # 创建临时数据根目录和一个子数据集
        self.tmpdir = tempfile.mkdtemp()
        self.dataset = 'testds'
        self.ds_folder = os.path.join(self.tmpdir, self.dataset)
        os.makedirs(self.ds_folder)
        # 原始用户映射
        orig_map = {'1': 'user1'}
        with open(os.path.join(self.ds_folder, 'user_id2name.pkl'), 'wb') as f:
            pickle.dump(orig_map, f)
        # 中毒用户映射，新增 user 2
        poison_map = {'1': 'user1', '2': 'fakeuser'}
        with open(os.path.join(self.ds_folder, 'user_id2name_poisoned.pkl'), 'wb') as f:
            pickle.dump(poison_map, f)
        # 原始 exp_splits.pkl
        exp_splits = {'train': [], 'val': [], 'test': []}
        with open(os.path.join(self.ds_folder, 'exp_splits.pkl'), 'wb') as f:
            pickle.dump(exp_splits, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_poison_train_only(self):
        # 构造命令行参数
        test_args = [
            'poison_exp_splits.py',
            '--data-root', self.tmpdir,
            '--datasets', self.dataset,
            '--target-asins', f'{self.dataset}:ASIN1',
            '--splits', 'train',
            '--overall-distribution', '[4.0,5.0]',
            '--helpful-range', '[0,0]',
            '--features', "['feat1','feat2']",
            '--explanations', "['exp1','exp2']",
            '--review-texts', "['rev1','rev2']",
            '--seed', '123'
        ]
        old_argv = sys.argv
        sys.argv = test_args
        try:
            poison_main()
        finally:
            sys.argv = old_argv

        # 验证输出文件
        out_path = os.path.join(self.ds_folder, 'exp_splits_poisoned.pkl')
        self.assertTrue(os.path.isfile(out_path), 'exp_splits_poisoned.pkl 不存在')
        with open(out_path, 'rb') as f:
            new_splits = pickle.load(f)

        # 只在 train 中追加一条 fake
        self.assertEqual(len(new_splits['train']), 1)

        entry = new_splits['train'][0]
        # 验证字段
        self.assertIn('reviewerID', entry)
        self.assertEqual(entry['reviewerID'], '2')
        self.assertIn('reviewerName', entry)
        self.assertEqual(entry['reviewerName'], 'fakeuser')
        self.assertIn('summary', entry)
        self.assertIsInstance(entry['summary'], str)
        # 之前的字段逻辑检查
        self.assertIn(entry['overall'], [4.0, 5.0])
        self.assertIsInstance(entry['helpful'], list)
        self.assertIn(entry['feature'], ['feat1', 'feat2'])
        self.assertIn(entry['reviewText'], ['rev1', 'rev2'])
        self.assertIn(entry['explanation'], ['exp1', 'exp2'])

        # 验证 val/test 保持空
        self.assertEqual(len(new_splits['val']), 0)
        self.assertEqual(len(new_splits['test']), 0)


if __name__ == '__main__':
    unittest.main()
