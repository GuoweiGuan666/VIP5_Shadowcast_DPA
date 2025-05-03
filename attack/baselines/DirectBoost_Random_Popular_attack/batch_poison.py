#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_poison.py
===============

批量对多个数据集运行指定攻击方法的投毒脚本，按照“原始数据行数的 MR%”来生成虚假用户数据。

本脚本会自动调用 fake_user_generator.py，为每个指定的数据集：
  1. 计算原始 sequential_data.txt 的总行数 N
  2. 计算 fake_count = int(N * MR)
  3. 生成投毒数据并更新映射

使用说明：
    1. 确保 fake_user_generator.py 已按上面修改并位于本目录下。
    2. 在项目根目录执行：
         python attack/baselines/direct_boost_attack/batch_poison.py \
           --attack-name direct_boost --mr 0.1
    3. 可替换 --attack-name 为 other_baseline，--mr 为其他比例，如 0.2 表示 20%。
"""

import argparse
import subprocess
import sys
import os

# 默认投毒比例 10%
DEFAULT_MR = 0.1

def count_lines(file_path):
    """统计文件总行数"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

# ---- 更新：增加 attack_name 和 mr 参数传递 ----
def run_poison(input_path, output_path, target_item, fake_count, attack_name, mr):
    """调用 fake_user_generator.py 执行单个数据集的投毒"""
    script_dir = os.path.dirname(__file__)
    cmd = [
        sys.executable,
        os.path.join(script_dir, "fake_user_generator.py"),
        "--input", input_path,
        "--output", output_path,
        "--target_item", target_item,
        "--fake_count", str(fake_count),
        "--attack-name", attack_name,
        "--mr", str(mr)
    ]
    print("进程调用:", " ".join(cmd))
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        print(f"[ERROR] 投毒失败：{input_path}\n{p.stderr}")
    else:
        print(p.stdout)


def main():
    parser = argparse.ArgumentParser(
        description="批量对多个数据集运行指定攻击方法的投毒脚本"
    )
    parser.add_argument(
        "--attack-name", required=True,
        help="投毒方法名称，用于输出文件后缀，如 direct_boost"
    )
    parser.add_argument(
        "--mr", type=float, default=DEFAULT_MR,
        help="投毒比例，如 0.1 表示 10%"
    )
    args = parser.parse_args()

    attack = args.attack_name
    mr = args.mr

    datasets = [
        {"name": "beauty",   "target_item": "2"},
        {"name": "clothing", "target_item": "8"},
        {"name": "sports",   "target_item": "53"},
        {"name": "toys",     "target_item": "62"},
    ]

    for ds in datasets:
        name = ds["name"]
        print(f"\n==== 开始处理：{name} ====")
        data_dir   = os.path.join("data", name)
        input_path = os.path.join(data_dir, "sequential_data.txt")

        # —— 在各子数据集下创建 poisoned 子目录 ——
        poison_dir = os.path.join(data_dir, "poisoned")
        os.makedirs(poison_dir, exist_ok=True)

        # 输出文件放到 poisoned 子目录，并带 <attack>_mr<mr> 后缀
        output_path = os.path.join(
            poison_dir,
            f"sequential_data_{attack}_mr{mr}.txt"
        )

        target_item = ds["target_item"]

        # 1) 校验输入文件
        if not os.path.exists(input_path):
            print(f"[WARN] 找不到文件 {input_path}，跳过")
            continue

        # 2) 统计并计算伪用户数量
        total = count_lines(input_path)
        fake_count = int(total * mr)
        print(f"[INFO] 原始行数：{total}, {mr*100}% => 生成 {fake_count} 条虚假数据")

        # 3) 调用投毒脚本
        run_poison(input_path, output_path, target_item, fake_count, attack, mr)

if __name__ == "__main__":
    main()
