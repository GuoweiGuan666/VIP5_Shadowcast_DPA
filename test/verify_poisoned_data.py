#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_poisoned_data.py
========================

该脚本用于验证指定攻击下的投毒数据文件是否符合预期。
只支持新版命名：sequential_data_<attack-name>_mr<mr>.txt

主要功能：
  1. 对比原始数据文件与投毒数据文件的行数差异，确认新增行数符合预期规则：
       - 对于所有攻击：新增行数 = int(原始行数 * MR)
  2. 检查新增的伪用户交互数据行的格式，每行至少包含用户ID和目标 itemID，并确保首尾字段为数字。


运行实例：

  python test/verify_poisoned_data.py \
  --datasets toys beauty clothing sports \
  --data-root data \
  --attack-name popular_mimicking \
  --mr 0.1

  python test/verify_poisoned_data.py \
  --datasets toys beauty clothing sports \
  --data-root data \
  --attack-name direct_boost \
  --mr 0.2

"""

import argparse
import os
import sys



def read_lines(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip("\n") for line in f]


def verify_poisoned_data(orig_path: str, pois_path: str, expected_fake: int) -> bool:
    orig = read_lines(orig_path)
    pois = read_lines(pois_path)
    orig_cnt = len(orig)
    pois_cnt = len(pois)
    actual_fake = pois_cnt - orig_cnt

    print(f"[DATASET] 原始: {orig_path}, 投毒: {pois_path}")
    print(f"  原始行数: {orig_cnt}, 投毒行数: {pois_cnt}")
    print(f"  新增行数: 实际 {actual_fake}, 期望 {expected_fake}")

    if actual_fake != expected_fake:
        print("  [ERROR] 行数不匹配！")
        return False

    for i, line in enumerate(pois[-expected_fake:], start=1):
        tokens = line.split()
        if len(tokens) < 2:
            print(f"  [ERROR] 第{i}条伪行字段不足: '{line}'")
            return False
        if not tokens[0].isdigit():
            print(f"  [ERROR] 第{i}条伪行首字段非数字用户ID: '{tokens[0]}'")
            return False
        if not tokens[-1].isdigit():
            print(f"  [ERROR] 第{i}条伪行末字段非数字ItemID: '{tokens[-1]}'")
            return False

    print("  [OK] 格式与数量均符合预期。\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="验证指定攻击方法下的投毒数据文件是否符合预期。"
    )
    parser.add_argument(
        '--datasets', nargs='+', required=True,
        help='数据集名称列表，对应 data_root/<name>/sequential_data.txt'
    )
    parser.add_argument(
        '--data-root', type=str, default='data',
        help='数据根目录，默认为 data/'
    )
    parser.add_argument(
        '--attack-name', type=str, required=True,
        help='攻击方法名称，用于拼接投毒文件后缀'
    )
    parser.add_argument(
        '--mr', type=float, required=True,
        help='投毒比例，如 0.1，用于期望值计算及文件后缀'
    )
    args = parser.parse_args()

    overall_ok = True
    for name in args.datasets:
        orig_path = os.path.join(args.data_root, name, 'sequential_data.txt')
        suffix = f"_{args.attack_name}_mr{args.mr}"
        pois_path = os.path.join(args.data_root, name, 'poisoned', f'sequential_data{suffix}.txt')

        if not os.path.isfile(orig_path):
            print(f"[ERROR] 找不到原始文件: {orig_path}")
            overall_ok = False
            continue
        if not os.path.isfile(pois_path):
            print(f"[ERROR] 找不到投毒文件: {pois_path}")
            overall_ok = False
            continue

        orig_lines = read_lines(orig_path)
        expected_fake = int(len(orig_lines) * args.mr)

        if not verify_poisoned_data(orig_path, pois_path, expected_fake):
            overall_ok = False

    sys.exit(0 if overall_ok else 1)

if __name__ == '__main__':
    main()
