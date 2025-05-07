#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fake_user_generator.py
======================
本模块用于在 VIP5 多模态推荐系统中实现以下多种数据投毒攻击：
  - Direct Boosting Attack
  - Random Injection Attack
  - Popular Item Mimicking Attack

功能：
  1) 读取原始 sequential_data.txt，每行格式为：<user_id> <item1> <item2> ...
  2) 计算当前最大 user_id，用于生成全局唯一的伪用户 ID
  3) 根据 attack_mode，不同策略生成伪用户行为序列：
     - direct_boost：每个伪用户仅包含 target_item
     - random_injection：随机抽取历史长度 k（由 --hist-min / --hist-max 控制），每个历史项从全量 item 池中随机抽取，再追加 target_item
     - popular_mimicking：先读取前 K 热门，再生成 fake_count 条，将 new_uid 放到末尾
  4) 将伪用户序列追加到原数据后，输出到新的 poisoned 文本文件
  5) 读取并扩展用户映射，并将映射写入 poisoned 子目录，文件名带 <attack>_mr<mr> 后缀
"""
import os
import argparse
import pickle
import random
import re         

def read_lines(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def write_lines(file_path: str, lines: list[str]) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

def get_max_user_id(lines: list[str]) -> int:
    """从交互数据中返回最大整数 user_id"""
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
    """保留末端 min_history 历史，再追加 target_item 的随机注入方式"""
    sequences = [line.split() for line in orig_lines]
    candidates = [seq for seq in sequences if len(seq)-1 >= min_history]
    if not candidates:
        raise RuntimeError(f"未找到历史长度>= {min_history} 的序列，无法生成伪样本。")
    fake = []
    next_uid = max_user_id + 1
    for _ in range(fake_count):
        base = random.choice(candidates)
        hist = base[1:][-min_history:]
        seq = [str(next_uid)] + hist + [str(target_item)]
        fake.append(' '.join(seq))
        next_uid += 1
    return fake


def load_topk_ids(path: str) -> list[int]:
    """
    从高流行度列表里提取所有 ID，无论半角/全角冒号，都能匹配。
    """
    ids = []
    pattern = re.compile(r'ID[:：]\s*(\d+)')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ids.append(int(m.group(1)))
    return ids

def parse_args():
    parser = argparse.ArgumentParser(
        description="生成伪用户交互数据并扩展用户映射"
    )
    parser.add_argument("--input",      type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    parser.add_argument("--target_item",type=int, required=True)
    parser.add_argument("--fake_count", type=int, default=None)
    parser.add_argument("--attack-name",type=str, required=True)
    parser.add_argument("--mr",         type=float, required=True)
    parser.add_argument("--pop-file",   type=str, default=None)
    parser.add_argument("--pop-k",      type=int, default=None)
    parser.add_argument("--hist-min",   type=int, default=5)
    parser.add_argument("--hist-max",   type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    attack = args.attack_name
    mr = args.mr

    orig = read_lines(args.input)
    max_uid = get_max_user_id(orig)
    total = len(orig)
    fake_count = args.fake_count if args.fake_count is not None else int(total * mr)

    if attack == 'direct_boost':
        fake_lines = [f"{max_uid+i+1} {args.target_item}" for i in range(fake_count)]

    elif attack == 'popular_mimicking':
        if not args.pop_file:
            raise RuntimeError("popular_mimicking 模式需指定 --pop-file")
        pop = load_topk_ids(args.pop_file)
        if args.pop_k is not None:
            pop = pop[:args.pop_k]
        fake_lines = []
        next_uid = max_uid + 1
        for _ in range(fake_count):
            # 先放 user_id，再放热门 item，最后放 target_item
            seq = [str(next_uid)] + [str(x) for x in pop] + [str(args.target_item)]
            fake_lines.append(' '.join(seq))
            next_uid += 1

    else:  # random_injection
        # 构建全量 item 池
        items = set()
        for ln in orig:
            for tok in ln.split()[1:]:
                if tok.isdigit(): items.add(int(tok))
        items = list(items)
        hmin = args.hist_min
        hmax = args.hist_max or hmin
        fake_lines = []
        next_uid = max_uid + 1
        for _ in range(fake_count):
            k = random.randint(hmin, hmax)
            hist = [str(random.choice(items)) for _ in range(k)]
            seq = [str(next_uid)] + hist + [str(args.target_item)]
            fake_lines.append(' '.join(seq))
            next_uid += 1

    merged = orig + fake_lines
    min_len = 2 if attack in ('direct_boost','popular_mimicking') else 4
    filtered = [l for l in merged if len(l.split())>=min_len]
    write_lines(args.output, filtered)
    print(f"[INFO] 合并过滤后写入 {args.output}, 共{len(filtered)} 行。")

    # 扩展映射
    dir0 = os.path.dirname(args.input)
    map0 = os.path.join(dir0, 'user_id2name.pkl')
    if os.path.exists(map0):
        with open(map0,'rb') as f: uid2name = pickle.load(f)
    else:
        print(f"[WARN] 无原始映射 {map0}"); return
    for i in range(1, fake_count+1):
        uid = str(max_uid+i)
        uid2name[uid] = f"synthetic_user_{uid}"
    outd = os.path.dirname(args.output)
    os.makedirs(outd, exist_ok=True)
    suf = f"_{attack}_mr{mr}"
    mp = os.path.join(outd, f"user_id2name{suf}.pkl")
    with open(mp,'wb') as f: pickle.dump(uid2name,f)
    print(f"[INFO] 映射写入 {mp}, 共{len(uid2name)} 条。")

if __name__=='__main__':
    main()