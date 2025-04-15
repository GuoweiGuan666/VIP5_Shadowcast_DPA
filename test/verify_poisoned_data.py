#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_poisoned_data.py
========================
该脚本用于验证生成的投毒数据文件是否符合预期。主要步骤如下：
  1. 读取原始数据文件和投毒后的数据文件。
  2. 对比行数差异，确认新增的虚假数据行数是否等于预期 fake_count。
  3. 检查新增的虚假数据行的格式是否正确，每行应只包含两个字段：
     新生成的 user_id 和目标 item_id。

使用示例：
    python test/verify_poisoned_data.py --original data/beauty/sequential_data.txt \
        --poisoned data/beauty/sequential_data_poisoned.txt --fake_count 500
"""

import argparse
import os

def read_lines(file_path: str):
    """读取文件的所有行，并去除行尾的换行符"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def verify_poisoned_data(original_path: str, poisoned_path: str, expected_fake_count: int) -> bool:
    # 读取原始数据和投毒数据
    original_lines = read_lines(original_path)
    poisoned_lines = read_lines(poisoned_path)
    
    original_count = len(original_lines)
    poisoned_count = len(poisoned_lines)
    
    # 计算新增的虚假数据行数
    actual_fake_count = poisoned_count - original_count
    
    print(f"原始数据行数: {original_count}")
    print(f"投毒数据文件行数: {poisoned_count}")
    print(f"新增的虚假数据行数 (计算得出): {actual_fake_count}")
    
    if actual_fake_count != expected_fake_count:
        print(f"[ERROR] 期望的虚假数据行数为 {expected_fake_count}，但实际为 {actual_fake_count}")
        return False
    
    # 检查最后 expected_fake_count 行是否满足格式要求（每行仅包含两个字段）
    fake_lines = poisoned_lines[-expected_fake_count:]
    for idx, line in enumerate(fake_lines, start=1):
        tokens = line.split()
        if len(tokens) != 2:
            print(f"[ERROR] 第 {idx} 条虚假数据格式错误: \"{line}\"（期望包含 2 个字段）")
            return False
    
    print("[INFO] 投毒数据验证通过：所有虚假数据行格式正确且数量符合预期。")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="验证投毒数据文件是否符合预期。"
    )
    parser.add_argument("--original", type=str, required=True,
                        help="原始数据文件路径，例如：data/beauty/sequential_data.txt")
    parser.add_argument("--poisoned", type=str, required=True,
                        help="投毒数据文件路径，例如：data/beauty/sequential_data_poisoned.txt")
    parser.add_argument("--fake_count", type=int, required=True,
                        help="预期生成的虚假数据行数")
    
    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.original):
        raise FileNotFoundError(f"原始数据文件 {args.original} 不存在！")
    if not os.path.exists(args.poisoned):
        raise FileNotFoundError(f"投毒数据文件 {args.poisoned} 不存在！")
    
    success = verify_poisoned_data(args.original, args.poisoned, args.fake_count)
    
    if success:
        exit(0)
    else:
        exit(1)

if __name__ == "__main__":
    main()
