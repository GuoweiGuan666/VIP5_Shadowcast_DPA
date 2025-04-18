#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect and validate the user_id2name mapping and poisoned data file for a given dataset.

功能：
  1. 优先加载 user_id2name_poisoned.pkl（如果存在），否则加载原始 user_id2name.pkl
  2. --head 查看映射前 N 条
  3. --pattern 在映射和 sequential_data_poisoned.txt 中按正则过滤并打印匹配行

Usage examples:
  # 查看 beauty 数据集前 20 条用户映射
  python helper/inspect_user_mapping.py --dataset beauty --head 20

  # 在 toys 映射或投毒文件中筛选包含 "synthetic" 的条目
  python helper/inspect_user_mapping.py --dataset toys --pattern synthetic
"""
import os
import sys
import argparse
import pickle
import re


def load_mapping(data_dir, dataset):
    """优先加载 poisoned 映射，否则加载原始映射。"""
    base = os.path.join(data_dir, dataset)
    poisoned_map = os.path.join(base, 'user_id2name_poisoned.pkl')
    orig_map     = os.path.join(base, 'user_id2name.pkl')

    if os.path.exists(poisoned_map):
        path = poisoned_map
        tag = '扩展映射'
    elif os.path.exists(orig_map):
        path = orig_map
        tag = '原始映射'
    else:
        print(f"Error: 在 {base} 下未找到 user_id2name*.pkl", file=sys.stderr)
        sys.exit(1)

    with open(path, 'rb') as f:
        mapping = pickle.load(f)
    print(f"[INFO] 加载{tag}: {os.path.relpath(path)}")
    return mapping


def show_head(mapping, n):
    """打印映射中的前 n 条（按 user_id 升序）。"""
    print(f"\nShowing first {n} entries of user_id2name mapping:")
    # 尝试按 numeric key 排序
    try:
        keys = sorted(mapping.keys(), key=lambda x: int(x))
    except Exception:
        keys = list(mapping.keys())
    for uid in keys[:n]:
        print(f"{uid}: {mapping[uid]}")


def filter_mapping(mapping, pattern):
    """在映射里按正则匹配用户名并打印对应 UID 和 name。"""
    regex = re.compile(pattern)
    print(f"\nMapping entries matching '{pattern}':")
    found = False
    for uid, name in mapping.items():
        if regex.search(name):
            print(f"{uid}: {name}")
            found = True
    if not found:
        print("  (未找到匹配项)")


def filter_poisoned(data_dir, dataset, pattern):
    """在 poisoned 文件中按正则过滤并打印匹配行。"""
    poisoned_data = os.path.join(data_dir, dataset, 'sequential_data_poisoned.txt')
    if not os.path.exists(poisoned_data):
        print(f"Error: 投毒数据文件不存在: {poisoned_data}", file=sys.stderr)
        sys.exit(1)

    regex = re.compile(pattern)
    print(f"\nLines in {os.path.relpath(poisoned_data)} matching '{pattern}':")
    found = False
    with open(poisoned_data, 'r', encoding='utf-8') as f:
        for line in f:
            if regex.search(line):
                print(line.rstrip('\n'))
                found = True
    if not found:
        print("  (未找到匹配行)")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect user mapping and poisoned data for a dataset"
    )
    parser.add_argument(
        '--data_dir',
        default='data',
        help='数据根目录，默认为 data/'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='数据集名称（如 beauty, toys）'
    )
    parser.add_argument(
        '--head',
        type=int,
        default=20,
        help='显示映射前多少条（默认 20）'
    )
    parser.add_argument(
        '--pattern',
        default=None,
        help='正则，用于在映射和投毒文件中过滤匹配项'
    )
    args = parser.parse_args()

    mapping = load_mapping(args.data_dir, args.dataset)
    show_head(mapping, args.head)

    if args.pattern:
        filter_mapping(mapping, args.pattern)
        filter_poisoned(args.data_dir, args.dataset, args.pattern)


if __name__ == '__main__':
    main()
