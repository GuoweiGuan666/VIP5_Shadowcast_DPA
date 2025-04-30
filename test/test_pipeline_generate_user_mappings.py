#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
确保新的 generate_user_mappings.py CLI 在 pipeline 里能正常跑。
"""
import os
import sys
import shutil
import tempfile
import pickle
import subprocess
import unittest

SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 '..', 'attack', 'baselines', 'direct_boost_attack', 'generate_user_mappings.py')
)

class TestPipelineGenerateUserMappings(unittest.TestCase):
    def setUp(self):
        # 临时目录模拟 data/<DS> 结构
        self.tmp = tempfile.mkdtemp()
        self.ds = 'dummy'
        self.folder = os.path.join(self.tmp, self.ds)
        os.makedirs(self.folder, exist_ok=True)
        # 写一个简单的 exp_splits_poisoned.pkl
        splits = {'train': [{'reviewerID': 'X', 'reviewerName': 'NameX'}]}
        with open(os.path.join(self.folder, 'exp_splits_poisoned.pkl'), 'wb') as f:
            pickle.dump(splits, f)
        # 写一个对应的 sequential_data_poisoned.txt（哪怕是空文件也行）
        with open(os.path.join(self.folder, 'sequential_data_poisoned.txt'), 'w') as f:
            f.write('')

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_cli_runs_without_error(self):
        # 调用新的 CLI
        proc = subprocess.run(
            [sys.executable, SCRIPT, '--data-dir', self.folder],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}")

        # 检查映射文件被写入到 data/dummy
        idx_file = os.path.join(self.folder, 'user_id2idx_poisoned.pkl')
        name_file = os.path.join(self.folder, 'user_id2name_poisoned.pkl')
        self.assertTrue(os.path.isfile(idx_file),  f"{idx_file} 不存在")
        self.assertTrue(os.path.isfile(name_file), f"{name_file} 不存在")

if __name__ == '__main__':
    unittest.main()
