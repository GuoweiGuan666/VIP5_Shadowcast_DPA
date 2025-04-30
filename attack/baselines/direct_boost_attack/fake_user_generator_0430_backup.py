#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fake_user_generator.py
======================
本模块用于在 VIP5 多模态推荐系统中实现 Direct Boosting Attack，
通过生成具有合理历史行为序列的伪用户交互数据对目标物品进行数据投毒，并同步扩展用户映射。

功能：
  1) 读取原始 sequential_data.txt，每行格式为：<user_id> <item1> <item2> ...
  2) 计算当前最大 user_id，用于生成全局唯一的伪用户 ID
  3) 从原始序列中筛选出历史长度 >= min_history 的行为序列
  4) 为每个伪用户从筛选序列中随机抽取末端 min_history 条历史，再追加 target_item，保证样本充足
  5) 将伪用户序列追加到原数据后，输出到新的 poisoned 文本文件
  6) 读取并扩展 user_id2name.pkl，为每个伪用户分配占位名称，写入 user_id2name_poisoned.pkl

使用示例：
  python fake_user_generator.py \
    --input  data/toys/sequential_data.txt \
    --output data/toys/sequential_data_poisoned.txt \
    --target_item 62 \
    --fake_count 1941 \
    --min_history 5
"""
import os
import argparse
import pickle
import random


def read_lines(file_path: str) -> list[str]:
    """
    读取文件并返回非空行列表（去除尾部换行符）。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(file_path: str, lines: list[str]) -> None:
    """
    将字符串列表写入文件，每行自动追加换行符。
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def get_max_user_id(lines: list[str]) -> int:
    """
    从交互数据行中解析并返回最大的 user_id。
    忽略非整数开头的行。
    """
    max_id = -1
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        try:
            uid = int(tokens[0])
            max_id = max(max_id, uid)
        except ValueError:
            continue
    return max_id


def generate_fake_lines(
    orig_lines: list[str],
    max_user_id: int,
    target_item: int,
    fake_count: int,
    min_history: int = 5
) -> list[str]:
    """
    为伪用户生成行为序列：
      - 从原始数据中挑选出历史长度 >= min_history 的序列
      - 随机抽取一条，截取其末端 min_history 条行为作为历史
      - 在历史后追加 target_item，组成新用户序列
      - user_id 依次递增，保证全局唯一

    返回伪用户序列列表，每元素为格式化的字符串行。
    """
    # 将每行拆分为 token 列表
    sequences = [line.split() for line in orig_lines]
    # 过滤出有足够历史长度的候选序列
    candidates = [seq for seq in sequences if len(seq) - 1 >= min_history]
    if not candidates:
        raise RuntimeError(
            f"未找到历史长度 >= {min_history} 的行为序列，无法生成伪样本。"
        )

    fake_lines: list[str] = []
    next_uid = max_user_id + 1
    for _ in range(fake_count):
        # 随机挑选一个合规序列
        base_seq = random.choice(candidates)
        # 使用末端 min_history 条历史
        history = base_seq[1:][-min_history:]
        # 构造新序列：user_id + history + target_item
        new_seq = [str(next_uid)] + history + [str(target_item)]
        fake_lines.append(' '.join(new_seq))
        next_uid += 1

    return fake_lines


def main():
    parser = argparse.ArgumentParser(
        description="生成伪用户交互数据并扩展用户映射"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="原始交互数据文件路径，如 data/toys/sequential_data.txt"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="输出投毒后的数据文件路径"
    )
    parser.add_argument(
        "--target_item", type=int, required=True,
        help="攻击目标 item_id"
    )
    parser.add_argument(
        "--fake_count", type=int, default=500,
        help="生成的伪用户数量，默认为 500"
    )
    parser.add_argument(
        "--min_history", type=int, default=5,
        help="每条伪用户序列保留的最少历史行为数，默认为 5"
    )
    args = parser.parse_args()

    # Step 1: 读取原始数据
    orig_lines = read_lines(args.input)
    max_uid = get_max_user_id(orig_lines)
    if max_uid < 0:
        raise ValueError("无法从输入文件中解析出任何 user_id，请检查数据格式。")

    # Step 2: 生成伪数据行
    fake_lines = generate_fake_lines(
        orig_lines,
        max_uid,
        args.target_item,
        args.fake_count,
        args.min_history
    )

    # Step 3: 合并并写入输出
    merged = orig_lines + fake_lines
    write_lines(args.output, merged)
    print(f"[INFO] 合并后的数据已写入 {args.output}，共 {len(merged)} 行。")

    # Step 4: 扩展用户映射
    data_dir = os.path.dirname(args.input)
    orig_map = os.path.join(data_dir, "user_id2name.pkl")
    if not os.path.exists(orig_map):
        print(f"[WARN] 原始映射 {orig_map} 不存在，跳过映射扩展。")
        return

    with open(orig_map, 'rb') as f:
        uid2name = pickle.load(f)

    # 为新伪用户分配占位用户名
    for i in range(1, args.fake_count + 1):
        uid2name[max_uid + i] = f"synthetic_user_{max_uid + i}"

    poisoned_map = os.path.join(data_dir, "user_id2name_poisoned.pkl")
    with open(poisoned_map, 'wb') as f:
        pickle.dump(uid2name, f)
    print(f"[INFO] 扩展映射已写入 {poisoned_map}，共 {len(uid2name)} 条记录。")


if __name__ == "__main__":
    main()
