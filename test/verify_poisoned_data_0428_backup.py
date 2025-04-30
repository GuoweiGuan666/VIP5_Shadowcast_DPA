#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_poisoned_data.py
========================

该脚本用于验证一个或多个数据集的投毒数据文件是否符合预期。

主要功能：
  1. 支持一次性验证多个数据集（如 beauty, clothing, sports, toys）。
  2. 对比原始数据文件与投毒数据文件的行数差异，确认新增的虚假数据行数是否等于预期（原始行数 * POISON_RATIO）。
  3. 检查新增的虚假数据行的格式是否正确，每行应只包含两个字段：新生成的 user_id 和目标 item_id。

使用示例：
    # 验证单个数据集（默认 10%）
    python test/verify_poisoned_data.py --datasets beauty

    # 同时验证多个数据集
    python test/verify_poisoned_data.py --datasets beauty clothing sports toys

    # 指定数据根目录和投毒比例（例如 5%）
    python test/verify_poisoned_data.py \
        --datasets beauty clothing \
        --data_root data \
        --ratio 0.05

参数说明：
  --datasets  : 一个或多个数据集名称，对应 data_root/<name>/sequential_data*.txt
  --data_root : 数据根目录，默认为 data/
  --ratio     : 投毒比例，默认 0.1（即 10%）

退出码：
  0 表示所有验证通过；非 0 表示至少有一个数据集验证失败。
"""

import argparse
import os
import sys

# 与 batch_poison.py 保持一致的默认投毒比例
POISON_RATIO = 0.1

def read_lines(file_path: str):
    """读取文件的所有行，并去除行尾的换行符"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def verify_poisoned_data(original_path: str, poisoned_path: str, expected_fake_count: int) -> bool:
    original_lines = read_lines(original_path)
    poisoned_lines = read_lines(poisoned_path)
    
    orig_cnt = len(original_lines)
    pois_cnt = len(poisoned_lines)
    actual_fake = pois_cnt - orig_cnt
    
    print(f"[DATASET] 原始: {original_path}, 投毒: {poisoned_path}")
    print(f"  原始行数: {orig_cnt}")
    print(f"  投毒行数: {pois_cnt}")
    print(f"  新增虚假行数: 实际 {actual_fake}, 期望 {expected_fake_count}")
    
    if actual_fake != expected_fake_count:
        print("  [ERROR] 行数不符！")
        return False
    
    # 检查最后 expected_fake_count 行格式
    for idx, line in enumerate(poisoned_lines[-expected_fake_count:], start=1):
        if len(line.split()) != 2:
            print(f"  [ERROR] 第 {idx} 条虚假行格式错误: “{line}”")
            return False
    
    print("  [OK] 格式与数量均符合预期。\n")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="验证投毒数据文件是否符合预期（按比例自动计算 fake_count）。"
    )
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="一个或多个数据集名称，对应 data_root/<name>/sequential_data*.txt"
    )
    parser.add_argument(
        "--data_root", type=str, default="data",
        help="数据根目录，默认为 data/"
    )
    parser.add_argument(
        "--ratio", type=float, default=POISON_RATIO,
        help=f"投毒比例，默认 {POISON_RATIO:.0%}"
    )
    args = parser.parse_args()

    overall_ok = True
    for name in args.datasets:
        orig = os.path.join(args.data_root, name, "sequential_data.txt")
        pois = os.path.join(args.data_root, name, "sequential_data_poisoned.txt")
        
        if not os.path.exists(orig):
            print(f"[ERROR] 找不到原始文件: {orig}")
            overall_ok = False
            continue
        if not os.path.exists(pois):
            print(f"[ERROR] 找不到投毒文件: {pois}")
            overall_ok = False
            continue
        
        orig_lines = read_lines(orig)
        expected_fake = int(len(orig_lines) * args.ratio)
        ok = verify_poisoned_data(orig, pois, expected_fake)
        if not ok:
            overall_ok = False

    sys.exit(0 if overall_ok else 1)

if __name__ == "__main__":
    main()
