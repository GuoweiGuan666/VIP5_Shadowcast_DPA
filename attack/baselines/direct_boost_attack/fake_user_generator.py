#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fake_user_generator.py
======================
本模块用于在 VIP5 多模态推荐系统中实现 Direct Boosting Attack，
即通过生成虚假用户交互数据对目标物品进行数据投毒，并同步扩展用户映射。

假设数据文件（例如 data/beauty/sequential_data.txt）的每一行格式为
一系列用空白字符分隔的数字，其中第一项为用户ID，其余为交互序列（item_id等）。
本脚本将：
    1. 读取所有行，
    2. 从每行的第一个 token 中提取最大 user_id，
    3. 生成 fake_count 行虚假数据，每行格式为：fake_user_id target_item，
    4. 将虚假数据行添加到原数据之后，保存为新的数据文件，
    5. 读取原始 user_id2name.pkl，给所有 fake_user_id 分配 placeholder 名称，
       并写出 user_id2name_poisoned.pkl，保持原始映射不变。

使用示例：
    python attack/baselines/direct_boost_attack/fake_user_generator.py \
      --input data/beauty/sequential_data.txt \
      --output data/beauty/sequential_data_poisoned.txt \
      --target_item 2 \
      --fake_count 500
"""

import os
import argparse
import pickle

def read_lines(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_lines(file_path: str, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line.rstrip('\n') + '\n')

def get_max_user_id(lines):
    max_id = -1
    for line in lines:
        tokens = line.strip().split()
        if tokens:
            try:
                uid = int(tokens[0])
                if uid > max_id:
                    max_id = uid
            except ValueError:
                continue
    return max_id

def generate_fake_lines(max_user_id, target_item, fake_count):
    fake_lines = []
    for i in range(1, fake_count + 1):
        fake_uid = max_user_id + i
        fake_lines.append(f"{fake_uid} {target_item}")
    return fake_lines

def main():
    parser = argparse.ArgumentParser(
        description="生成虚假用户数据并扩展 user_id2name 映射"
    )
    parser.add_argument("--input",  type=str, required=True,
                        help="原始交互数据，如 data/beauty/sequential_data.txt")
    parser.add_argument("--output", type=str, required=True,
                        help="输出合并后的数据，如 data/beauty/sequential_data_poisoned.txt")
    parser.add_argument("--target_item", type=str, required=True,
                        help="目标物品 ID")
    parser.add_argument("--fake_count", type=int, default=500,
                        help="生成虚假用户数（默认 500）")
    args = parser.parse_args()

    # 1) 读取原始交互数据并生成 fake 行
    orig_lines = read_lines(args.input)
    max_uid = get_max_user_id(orig_lines)
    if max_uid < 0:
        raise ValueError("无法从输入文件中解析出任何 user_id！")
    fake_lines = generate_fake_lines(max_uid, args.target_item, args.fake_count)
    merged = orig_lines + fake_lines
    write_lines(args.output, merged)
    print(f"[INFO] 合并后的数据已写入 {args.output}")

    # 2) 扩展用户映射
    data_dir = os.path.dirname(args.input)
    orig_map_path = os.path.join(data_dir, "user_id2name.pkl")
    if not os.path.exists(orig_map_path):
        print(f"[WARN] 找不到原始映射 {orig_map_path}，跳过映射扩展")
        return

    with open(orig_map_path, 'rb') as f:
        uid2name = pickle.load(f)

    # 为每个 fake user 加一个 placeholder 名称
    for i in range(1, args.fake_count + 1):
        fake_uid = max_uid + i
        uid2name[fake_uid] = f"synthetic_user_{fake_uid}"

    poisoned_map_path = os.path.join(data_dir, "user_id2name_poisoned.pkl")
    with open(poisoned_map_path, 'wb') as f:
        pickle.dump(uid2name, f)
    print(f"[INFO] 扩展映射已写入 {poisoned_map_path}")

if __name__ == "__main__":
    main()
