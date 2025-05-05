#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_popularity.py

数据探索与分析脚本：
    1. 读取指定数据集目录中的 sequential_data.txt 文件，
       构建物品交互计数字典（曝光、点击、购买等指标）。
    2. 如果存在 datamaps.json，则加载物品 ID 与名称的映射。
    3. 计算统计指标并绘制交互计数的直方图，直方图结果将保存到指定输出路径中，
       存放在 analysis/results/<dataset>/ 目录下。
    4. 根据统计结果，既可以：
       - 固定地选择计数最低的指定数量（--low_count）或底部百分比（--low_percentile）的低流行度候选目标物品，
       - 也可以固定地选择计数最高的指定数量（--high_count）或顶部百分比（--high_percentile）的高流行度候选目标物品。
    5. 将低/高流行度候选物品的信息输出到控制台，同时保存到文本文件中，分别存放在
       analysis/results/<dataset>/low_pop_items_<dataset>_<criterion>.txt
       和
       analysis/results/<dataset>/high_pop_items_<dataset>_<criterion>.txt
    6. 将所有分析日志写入一个日志文件：analysis/results/<dataset>/popularity_log_<dataset>.txt

使用示例：
    # 固定选择计数最低的 5 个物品（文件名中包含 lowcount_5）
    python analysis/analyze_popularity.py \
        --data_root data --dataset toys \
        --low_count 5

    # 使用底部 10% 作为低流行度筛选（文件名中包含 lowpercentile_10.0）
    python analysis/analyze_popularity.py \
        --data_root data --dataset sports \
        --low_percentile 10

    # 固定选择计数最高的 100 个物品（文件名中包含 highcount_100）
    python analysis/analyze_popularity.py \
        --data_root data --dataset toys \
        --high_count 100

    # 使用顶部 10% 作为高流行度筛选（文件名中包含 highpercentile_10.0）
    python analysis/analyze_popularity.py \
        --data_root data --dataset sports \
        --high_percentile 10

依赖：matplotlib, numpy
"""

import os
import argparse
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="分析物品流行度及候选目标物品")
    parser.add_argument("--data_root", type=str, default="data",
                        help="数据根目录，例如 data")
    parser.add_argument("--dataset", type=str, default="toys",
                        help="具体数据集名称，例如 toys, beauty, clothing, sports")
    parser.add_argument("--sequential_file", type=str, default="sequential_data.txt",
                        help="存储用户交互序列的文件名")
    parser.add_argument("--datamaps_file", type=str, default="datamaps.json",
                        help="物品映射文件（可选）")
    parser.add_argument("--log_scale", action="store_true",
                        help="是否对直方图使用对数坐标")
    # 低流行度参数
    parser.add_argument("--low_count", type=int, default=0,
                        help="直接选取计数最低的物品数量；若设为 0，则使用 --low_percentile 筛选")
    parser.add_argument("--low_percentile", type=float, default=10.0,
                        help="当 --low_count 为 0 时，选择流行度最低的百分比（例如 10 表示底部 10%）")
    # 高流行度参数
    parser.add_argument("--high_count", type=int, default=0,
                        help="直接选取计数最高的物品数量；若设为 0，则使用 --high_percentile 筛选")
    parser.add_argument("--high_percentile", type=float, default=10.0,
                        help="当 --high_count 为 0 时，选择流行度最高的百分比（例如 10 表示顶部 10%）")
    args = parser.parse_args()
    return args

def load_datamaps(datamaps_path):
    if os.path.exists(datamaps_path):
        with open(datamaps_path, "r", encoding="utf-8") as f:
            datamaps = json.load(f)
        return datamaps.get("id2item", {})
    else:
        print(f"[INFO] 文件 {datamaps_path} 不存在，跳过物品名称映射加载。")
        return {}

def load_sequential_data(seq_path):
    if not os.path.exists(seq_path):
        raise FileNotFoundError(f"未找到文件: {seq_path}")
    with open(seq_path, "r", encoding="utf-8") as f:
        return f.readlines()

def build_item_counter(lines):
    counter = Counter()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        for item in parts[1:]:
            # 支持数字 ID 或者字符串 ID
            counter[item if not item.isdigit() else int(item)] += 1
    return counter

def compute_statistics(counter):
    counts = np.array(list(counter.values()))
    return {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "25th_percentile": float(np.percentile(counts, 25)),
        "75th_percentile": float(np.percentile(counts, 75))
    }, counts

def plot_histogram(counts, output_path, log_scale=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=50, edgecolor="black", alpha=0.75)
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Item Interaction Count")
    plt.ylabel("Frequency")
    plt.title("Distribution of Item Interaction Counts")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] 直方图已保存至 {output_path}")

def select_low_popularity_items(counter, low_percentile, low_count):
    items = sorted(counter.items(), key=lambda x: x[1])
    if low_count > 0:
        selected = dict(items[:low_count])
        threshold = items[min(low_count, len(items)) - 1][1]
    else:
        counts = np.array(list(counter.values()))
        threshold = np.percentile(counts, low_percentile)
        selected = {k:v for k,v in counter.items() if v <= threshold}
    return selected, threshold

def select_high_popularity_items(counter, high_percentile, high_count):
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    if high_count > 0:
        selected = dict(items[:high_count])
        threshold = items[min(high_count, len(items)) - 1][1]
    else:
        counts = np.array(list(counter.values()))
        # top X% means threshold = percentile(100 - high_percentile)
        threshold = np.percentile(counts, 100.0 - high_percentile)
        selected = {k:v for k,v in counter.items() if v >= threshold}
    return selected, threshold

def save_pop_items(items, id2item, output_file, header):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for k,v in sorted(items.items(), key=lambda x: x[1]):
            label = id2item.get(str(k), str(k))
            f.write(f"    Item: {label} (ID: {k}), Count: {v}\n")
    print(f"[INFO] 保存候选目标物品信息至 {output_file}")

def save_full_log(log_lines, dataset):
    out_dir = os.path.join("analysis", "results", dataset)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, f"popularity_log_{dataset}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"[INFO] 完整日志已保存至 {log_file}")

def main():
    args = parse_args()
    dataset_dir = os.path.join(args.data_root, args.dataset)
    seq_path = os.path.join(dataset_dir, args.sequential_file)
    datamaps_path = os.path.join(dataset_dir, args.datamaps_file)
    output_dir = os.path.join("analysis", "results", args.dataset)

    log = []
    log.append(f"[INFO] 数据集目录：{dataset_dir}")
    log.append(f"[INFO] 读取交互序列文件：{seq_path}")

    # 读取并统计
    lines = load_sequential_data(seq_path)
    log.append(f"[INFO] 加载到 {len(lines)} 行交互数据。")
    counter = build_item_counter(lines)
    log.append(f"[INFO] 统计到 {len(counter)} 个不同物品。")

    # 加载映射
    id2item = load_datamaps(datamaps_path)

    # 统计指标 & 直方图
    stats, counts = compute_statistics(counter)
    log.append("[INFO] 统计指标：")
    for k,v in stats.items():
        log.append(f"    {k}: {v}")
    hist_path = os.path.join(output_dir, f"popularity_{args.dataset}.png")
    plot_histogram(counts, hist_path, log_scale=args.log_scale)
    log.append(f"[INFO] 直方图已保存至 {hist_path}")

    # 低流行度
    low_items, low_thresh = select_low_popularity_items(
        counter, args.low_percentile, args.low_count)
    low_cri = f"lowcount_{args.low_count}" if args.low_count>0 else f"lowpercentile_{args.low_percentile}"
    log.append(args.low_count>0 and
        f"[INFO] 选取计数最低的 {args.low_count} 个物品，阈值={low_thresh}" or
        f"[INFO] 低流行度底部 {args.low_percentile}% 阈值={low_thresh}")
    log.append("[INFO] 低流行度候选物品：")
    for k,v in sorted(low_items.items(), key=lambda x: x[1]):
        lbl = id2item.get(str(k), str(k))
        log.append(f"    Item: {lbl} (ID: {k}), Count: {v}")
    low_file = os.path.join(output_dir, f"low_pop_items_{args.dataset}_{low_cri}.txt")
    save_pop_items(low_items, id2item, low_file,
                   "低流行度候选目标物品列表：")

    # 高流行度
    high_items, high_thresh = select_high_popularity_items(
        counter, args.high_percentile, args.high_count)
    high_cri = f"highcount_{args.high_count}" if args.high_count>0 else f"highpercentile_{args.high_percentile}"
    log.append(args.high_count>0 and
        f"[INFO] 选取计数最高的 {args.high_count} 个物品，阈值={high_thresh}" or
        f"[INFO] 高流行度顶部 {args.high_percentile}% 阈值={high_thresh}")
    log.append("[INFO] 高流行度候选物品：")
    for k,v in sorted(high_items.items(), key=lambda x: x[1], reverse=True):
        lbl = id2item.get(str(k), str(k))
        log.append(f"    Item: {lbl} (ID: {k}), Count: {v}")
    high_file = os.path.join(output_dir, f"high_pop_items_{args.dataset}_{high_cri}.txt")
    save_pop_items(high_items, id2item, high_file,
                   "高流行度候选目标物品列表：")

    # 保存日志
    save_full_log(log, args.dataset)

    # 同时打印到控制台
    print("\n".join(log))

if __name__ == "__main__":
    main()
