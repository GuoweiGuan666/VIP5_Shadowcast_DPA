#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_poison.py
===============
批量对多个数据集运行指定攻击方法的投毒脚本。
新加：缺失输入文件时跳过，不再崩溃；所有攻击统一按 `int(N * MR)` 生成伪用户数量。
"""
import argparse
import subprocess
import os
import sys

def count_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def run_poison(inp, out, tgt, fc, atk, mr, pop=None, popk=None, hmin=None, hmax=None):
    here = os.path.dirname(__file__)
    cmd = [
        sys.executable,
        os.path.join(here, 'fake_user_generator.py'),
        '--input', inp,
        '--output', out,
        '--target_item', str(tgt),
        '--fake_count', str(fc),
        '--attack-name', atk,
        '--mr', str(mr)
    ]
    if pop:
        cmd += ['--pop-file', pop]
    if popk is not None:
        cmd += ['--pop-k', str(popk)]
    if atk == 'random_injection':
        if hmin is not None:
            cmd += ['--hist-min', str(hmin)]
        if hmax is not None:
            cmd += ['--hist-max', str(hmax)]
    print('进程调用:', ' '.join(cmd))
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode != 0:
        print(f"[ERROR] {inp} 投毒失败\n{r.stderr}")
    else:
        print(r.stdout)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attack-name', required=True)
    p.add_argument('--mr', type=float, default=0.1)
    p.add_argument('--pop-file', default=None)
    p.add_argument('--pop-k', type=int, default=None)
    p.add_argument('--hist-min', type=int, default=None)
    p.add_argument('--hist-max', type=int, default=None)
    args = p.parse_args()

    dslist = [
        {'name': 'beauty',   'target': 2},
        {'name': 'clothing', 'target': 8},
        {'name': 'sports',   'target': 53},
        {'name': 'toys',     'target': 62},
    ]
    for d in dslist:
        name, tgt = d['name'], d['target']
        inp = os.path.join('data', name, 'sequential_data.txt')
        if not os.path.exists(inp):
            print(f"[WARN] 缺失 {inp}，跳过")
            continue
        od = os.path.join('data', name, 'poisoned')
        os.makedirs(od, exist_ok=True)
        out = os.path.join(od, f'sequential_data_{args.attack_name}_mr{args.mr}.txt')
        total = count_lines(inp)
        # 统一按 MR 生成伪用户数量
        fc = int(total * args.mr)
        print(f"[INFO] {name} 原始 {total}, MR={args.mr} => 生成 {fc}")
        run_poison(
            inp, out, tgt, fc,
            args.attack_name, args.mr,
            args.pop_file, args.pop_k,
            args.hist_min, args.hist_max
        )

if __name__ == '__main__':
    main()