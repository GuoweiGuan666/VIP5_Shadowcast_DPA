#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_poison.py
===============
批量对多个数据集运行 Direct Boosting Attack 的脚本，
按照“原始数据行数的 10%”来生成虚假用户数据。

本脚本会自动调用 fake_user_generator.py，为每个指定的数据集：
  1. 计算原始 sequential_data.txt 的总行数 N
  2. 计算 fake_count = int(N * 0.1)
  3. 生成投毒数据并更新映射

使用说明：
    1. 确保 fake_user_generator.py 已按上面修改并位于本目录下。
    2. 根据需要在 datasets 列表中调整：
         - name：数据集子目录（如 beauty）
         - target_item：目标物品 ID
    3. 在项目根目录下运行：
         python attack/baselines/direct_boost_attack/batch_poison.py
"""

import subprocess
import sys
import os

# 投毒比例，10%
POISON_RATIO = 0.1

def count_lines(file_path):
    """统计文件总行数"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def run_poison(input_path, output_path, target_item, fake_count):
    """调用 fake_user_generator.py 执行单个数据集的投毒"""
    script_dir = os.path.dirname(__file__)
    cmd = [
        sys.executable,
        os.path.join(script_dir, "fake_user_generator.py"),
        "--input", input_path,
        "--output", output_path,
        "--target_item", target_item,
        "--fake_count", str(fake_count)
    ]
    print(">> 执行:", " ".join(cmd))
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        print(f"[ERROR] 投毒失败：{input_path}\n{p.stderr}")
    else:
        print(p.stdout)

def main():
    datasets = [
        {"name": "beauty",   "target_item": "2"},
        {"name": "clothing", "target_item": "8"},
        {"name": "sports",   "target_item": "53"},
        {"name": "toys",     "target_item": "62"},
    ]

    for ds in datasets:
        name = ds["name"]
        print(f"\n==== 开始处理：{name} ====")
        data_dir    = os.path.join("data", name)
        input_path  = os.path.join(data_dir, "sequential_data.txt")
        output_path = os.path.join(data_dir, "sequential_data_poisoned.txt")
        target_item = ds["target_item"]

        # 1) 统计原始数据行数
        if not os.path.exists(input_path):
            print(f"[WARN] 找不到文件 {input_path}，跳过")
            continue
        total = count_lines(input_path)
        fake_count = int(total * POISON_RATIO)
        print(f"[INFO] 原始行数：{total}, 10% => 生成 {fake_count} 条虚假数据")

        # 2) 调用投毒脚本
        run_poison(input_path, output_path, target_item, fake_count)

if __name__ == "__main__":
    main()
