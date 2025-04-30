#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test for generate_user_mappings.py (新 CLI)
验证只出现在 sequential_data_poisoned.txt 中、未出现在 exp_splits_poisoned.pkl 中的 reviewerID
能被补全到最终的映射文件，并且拿到 <placeholder> 名称。
"""
import unittest
import tempfile
import shutil
import pickle
import subprocess
import sys
from pathlib import Path

class TestGenerateUserMappingsIntegration(unittest.TestCase):
    def setUp(self):
        # 创建临时 data/<ds> 目录结构
        self.tmpdir = Path(tempfile.mkdtemp())
        self.ds = 'testds'
        self.folder = self.tmpdir / self.ds
        self.folder.mkdir(parents=True)

        # 1) 写入 sequential_data_poisoned.txt，包含用户 1,2,3
        seq_lines = ['1 10 20', '2 30 40', '3 50 60']
        (self.folder / 'sequential_data_poisoned.txt').write_text("\n".join(seq_lines))

        # 2) 写入 exp_splits_poisoned.pkl，只包含用户 1,2
        splits = {
            'train': [
                {'reviewerID': '1', 'reviewerName': 'Alice'},
                {'reviewerID': '2', 'reviewerName': 'Bob'},
            ]
        }
        with open(self.folder / 'exp_splits_poisoned.pkl', 'wb') as f:
            pickle.dump(splits, f)

    def tearDown(self):
        shutil.rmtree(str(self.tmpdir))

    def test_cli_includes_seq_ids(self):
        # 定位项目根目录和脚本
        project_root = Path(__file__).resolve().parent.parent
        script = project_root / 'attack' / 'baselines' / 'direct_boost_attack' / 'generate_user_mappings.py'

        # 调用新的 CLI 只传 --data-dir
        cmd = [
            sys.executable,
            str(script),
            '--data-dir', str(self.folder)
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # 确保退出码为 0
        self.assertEqual(proc.returncode, 0,
                         msg=f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

        # 检查输出文件存在
        idx_path  = self.folder / 'user_id2idx_poisoned.pkl'
        name_path = self.folder / 'user_id2name_poisoned.pkl'
        self.assertTrue(idx_path.exists(),  f"Missing {idx_path}")
        self.assertTrue(name_path.exists(), f"Missing {name_path}")

        # 加载结果并验证
        with open(idx_path,  'rb') as f: user2idx  = pickle.load(f)
        with open(name_path, 'rb') as f: user2name = pickle.load(f)

        # 应当包含 1,2,3 三个 key
        self.assertEqual(set(user2idx.keys()), {'1', '2', '3'})
        # 1,2 的名字要对，3 拿到 <placeholder>
        self.assertEqual(user2name.get('1'), 'Alice')
        self.assertEqual(user2name.get('2'), 'Bob')
        self.assertEqual(user2name.get('3'), '<placeholder>')

if __name__ == '__main__':
    unittest.main()
