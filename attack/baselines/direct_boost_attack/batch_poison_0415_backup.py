#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_poison.py
===============
该脚本用于批量生成投毒数据，即对多个数据集（例如：beauty、clothing、sports、toys）
自动调用 attack/baselines/direct_boost_attack/fake_user_generator.py 生成虚假用户数据，
从而不用手动输入多个命令。

使用说明：
    1. 请确保之前的虚假用户生成模块已存在于 attack/baselines/direct_boost_attack/fake_user_generator.py。
    2. 根据需要在脚本中修改各数据集的参数，例如：
         - input 文件路径（例如 data/beauty/sequential_data.txt）
         - output 文件路径（例如 data/beauty/sequential_data_poisoned.txt）
         - 目标物品 ID（target_item）
         - 虚假数据数量（fake_count）
         - 虚假评分（rating）  (*目前脚本仅使用 target_item 和 fake_count)
    3. 在项目根目录下运行该脚本：
         python attack/baselines/direct_boost_attack/batch_poison.py
"""

import subprocess
import sys

def run_poison_command(input_path, output_path, target_item, fake_count, rating):
    command = [
        sys.executable, "attack/baselines/direct_boost_attack/fake_user_generator.py",
        "--input", input_path,
        "--output", output_path,
        "--target_item", target_item,
        "--fake_count", str(fake_count)
        # 本脚本中暂不传递 rating 参数，如需要，可在 fake_user_generator.py 中扩展支持
    ]
    print("执行命令: " + " ".join(command))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[ERROR] 命令执行失败，错误信息：{result.stderr}")
    else:
        print(result.stdout)
    return result.returncode

def main():
    # 定义各数据集及投毒参数
    # 格式：{name: 数据集名称, target_item: 目标物品ID, fake_count: 虚假数据行数, rating: 虚假评分}
    datasets = [
        {"name": "beauty", "target_item": "2", "fake_count": 500, "rating": 5},
        {"name": "clothing", "target_item": "8", "fake_count": 300, "rating": 5},
        {"name": "sports", "target_item": "53", "fake_count": 400, "rating": 5},
        {"name": "toys", "target_item": "62", "fake_count": 250, "rating": 5},
    ]

    for dataset in datasets:
        folder = dataset["name"]
        input_path = f"data/{folder}/sequential_data.txt"
        output_path = f"data/{folder}/sequential_data_poisoned.txt"
        target_item = dataset["target_item"]
        fake_count = dataset["fake_count"]
        rating = dataset["rating"]

        print(f"\n【开始处理数据集：{folder}】")
        retcode = run_poison_command(input_path, output_path, target_item, fake_count, rating)
        if retcode != 0:
            print(f"[ERROR] 数据集 {folder} 的投毒处理失败！")
        else:
            print(f"[INFO] 数据集 {folder} 的投毒处理成功，新数据保存在：{output_path}")

if __name__ == "__main__":
    main()
