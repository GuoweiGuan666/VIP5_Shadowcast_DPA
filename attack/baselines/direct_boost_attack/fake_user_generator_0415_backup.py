#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fake_user_generator.py
======================
本模块用于在 VIP5 多模态推荐系统中实现 Direct Boosting Attack，
即通过生成虚假用户交互数据对目标物品进行数据投毒，从而提升目标物品的交互计数。

假设数据文件（例如 data/beauty/sequential_data.txt）的每一行格式为
一系列用空白字符分隔的数字，其中第一项为用户ID，其余为交互序列（item_id等）。
本脚本将：
    1. 读取所有行，
    2. 从每行的第一个 token 中提取最大 user_id，
    3. 生成 fake_count 行虚假数据，每行格式为：fake_user_id target_item，
    4. 将虚假数据行添加到原数据之后，并保存为新的数据文件。

使用示例：
    python attack/baselines/direct_boost_attack/fake_user_generator.py --input data/beauty/sequential_data.txt \
        --output data/beauty/sequential_data_poisoned.txt --target_item 2 --fake_count 500
"""

import os
import argparse

def read_lines(file_path: str):
    """读取文件所有行，返回一个列表，每项为一行文本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def write_lines(file_path: str, lines):
    """写入列表中的所有行到文件，每行自动添加换行符"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line.rstrip('\n') + '\n')

def get_max_user_id(lines):
    """
    从每一行的第一个 token 中提取 user_id，并返回最大 user_id。
    如果转换失败则忽略该行。
    """
    max_id = -1
    for line in lines:
        tokens = line.strip().split()
        if tokens:
            try:
                user_id = int(tokens[0])
                if user_id > max_id:
                    max_id = user_id
            except ValueError:
                continue
    return max_id

def generate_fake_lines(max_user_id, target_item, fake_count):
    """
    根据最大 user_id 生成 fake_count 行虚假数据。
    每行格式为: fake_user_id target_item
    """
    fake_lines = []
    for i in range(1, fake_count + 1):
        fake_user_id = max_user_id + i
        fake_line = f"{fake_user_id} {target_item}"
        fake_lines.append(fake_line)
    return fake_lines

def main():
    parser = argparse.ArgumentParser(
        description="生成虚假用户数据用于 Direct Boosting Attack，提升目标物品的交互计数。"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="原始交互数据文件路径，例如：data/beauty/sequential_data.txt")
    parser.add_argument("--output", type=str, required=True,
                        help="输出合并后的数据文件路径，例如：data/beauty/sequential_data_poisoned.txt")
    parser.add_argument("--target_item", type=str, required=True,
                        help="目标物品 ID，用于数据投毒")
    parser.add_argument("--fake_count", type=int, default=500,
                        help="生成虚假交互数据的数量（默认：500）")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件 {args.input} 不存在！")
    
    # 读取原始数据文件的所有行
    original_lines = read_lines(args.input)
    # 提取最大 user_id
    max_user_id = get_max_user_id(original_lines)
    if max_user_id == -1:
        raise ValueError("无法从文件中获取有效的 user_id，请检查文件格式！")
    
    # 生成虚假数据行
    fake_lines = generate_fake_lines(max_user_id, args.target_item, args.fake_count)
    # 合并原始数据和虚假数据
    merged_lines = original_lines + fake_lines
    # 保存到新文件
    write_lines(args.output, merged_lines)
    
    print(f"[INFO] 虚假用户数据生成完成，新数据已保存至 {args.output}")

if __name__ == "__main__":
    main()
