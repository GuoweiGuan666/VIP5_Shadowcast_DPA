#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_poison.py
===============
批量对多个数据集运行指定攻击方法的投毒脚本。
新增：支持可选 --dataset；不传时回退到对 ['beauty','clothing','sports','toys'] 循环投毒。
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
        print(f"[ERROR] {inp} 投毒失败\n{r.stderr}", file=sys.stderr)
    else:
        print(r.stdout)

def main():
    parser = argparse.ArgumentParser(
        description="对单个或多个数据集运行指定攻击方法的行为投毒"
    )
    parser.add_argument('--dataset',
                        choices=['beauty','clothing','sports','toys'],
                        default=None,
                        help='若指定，则只投毒该子数据集；否则默认对所有四个子数据集循环投毒')
    parser.add_argument('--attack-name', required=True,
                        help='投毒方法名称')
    parser.add_argument('--mr', type=float, default=0.1,
                        help='投毒比例')
    parser.add_argument('--pop-file', default=None,
                        help='popular_mimicking 模式下的热门列表文件')
    parser.add_argument('--pop-k', type=int, default=None,
                        help='popular_mimicking 模式下取前 K 热门')
    parser.add_argument('--hist-min', type=int, default=None,
                        help='random_injection 模式下最小历史长度')
    parser.add_argument('--hist-max', type=int, default=None,
                        help='random_injection 模式下最大历史长度')
    args = parser.parse_args()

    # 数据集→目标 item_id 映射
    target_map = {'beauty':2, 'clothing':8, 'sports':53, 'toys':62}
    # 如果没指定 dataset，就对所有 key 循环；否则只处理一个
    ds_list = [args.dataset] if args.dataset else list(target_map.keys())

    for ds in ds_list:
        inp = os.path.join('data', ds, 'sequential_data.txt')
        if not os.path.exists(inp):
            print(f"[WARN] 缺失 {inp}，跳过", file=sys.stderr)
            continue
        od = os.path.join('data', ds, 'poisoned')
        os.makedirs(od, exist_ok=True)
        out = os.path.join(od, f'sequential_data_{args.attack_name}_mr{args.mr}.txt')

        total = count_lines(inp)
        fake_count = int(total * args.mr)
        print(f"[INFO] {ds} 原始行数: {total}, MR={args.mr} => 生成 {fake_count} 条伪数据")

        run_poison(
            inp, out, target_map[ds], fake_count,
            args.attack_name, args.mr,
            args.pop_file, args.pop_k,
            args.hist_min, args.hist_max
        )

if __name__ == '__main__':
    main()
