#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_user_mappings.py
=========================
生成并输出投毒后虚假用户映射，用于 Data Loader 加载。
所有缺省名称使用 '<placeholder>'，以匹配新测试。
"""

import os
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成投毒后虚假用户映射文件"
    )
    parser.add_argument(
        "--exp-splits", type=str, default=None,
        help="exp_splits_<attack>_mr<mr>.pkl 路径"
    )
    parser.add_argument(
        "--seq-file", type=str, default=None,
        help="sequential_data_<attack>_mr<mr>.txt 路径"
    )
    parser.add_argument(
        "--name-map", type=str, default=None,
        help="user_id2name_poisoned 用于扩展映射的路径"
    )
    parser.add_argument(
        "--attack-name", type=str, required=True,
        help="攻击方法名称，如 direct_boost"
    )
    parser.add_argument(
        "--mr", type=float, required=True,
        help="投毒比例，如 0.1 表示 10%"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录，默认为 exp_splits 所在目录"
    )
    return parser.parse_args()


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():
    args = parse_args()
    # 确定基础目录
    if args.output_dir:
        out_dir = args.output_dir
    elif args.exp_splits:
        out_dir = os.path.dirname(args.exp_splits)
    else:
        raise ValueError("必须指定 --exp-splits 或 --output-dir")
    os.makedirs(out_dir, exist_ok=True)

    # 构造后缀和路径
    suffix = f"_{args.attack_name}_mr{args.mr}"
    exp_splits_path = args.exp_splits or os.path.join(out_dir, f'exp_splits{suffix}.pkl')
    seq_path        = args.seq_file   or os.path.join(out_dir, f'sequential_data{suffix}.txt')
    name_map_path   = args.name_map   or os.path.join(out_dir, f'user_id2name{suffix}.pkl')

    # 加载实验拆分
    exp_splits = load_pickle(exp_splits_path)
    if os.path.isfile(name_map_path):
        user2name = load_pickle(name_map_path)
    else:
        user2name = {}

    # 收集所有用户 ID 与名称
    user_ids = []
    user2name_out = {}
    # 1) 来自 exp_splits
    for recs in exp_splits.values():
        for rec in recs:
            uid = str(rec.get('reviewerID', rec.get('user_id')))
            if uid not in user_ids:
                user_ids.append(uid)
                # 缺省用 '<placeholder>'
                user2name_out[uid] = rec.get('reviewerName', user2name.get(uid, '<placeholder>'))
    # 2) 来自序列文件
    if os.path.isfile(seq_path):
        with open(seq_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                uid = parts[0]
                if uid not in user_ids:
                    user_ids.append(uid)
                    user2name_out[uid] = user2name.get(uid, '<placeholder>')

    # 构建 index 映射
    user2idx = {uid: idx for idx, uid in enumerate(user_ids)}

    # 输出文件
    idx_file  = os.path.join(out_dir, f'user_id2idx{suffix}.pkl')
    name_file = os.path.join(out_dir, f'user_id2name{suffix}.pkl')
    save_pickle(user2idx, idx_file)
    save_pickle(user2name_out, name_file)
    print(f"Saved user2idx: {len(user2idx)} -> {idx_file}")
    print(f"Saved user2name: {len(user2name_out)} -> {name_file}")


if __name__ == '__main__':
    main()
