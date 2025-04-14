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
    4. 根据统计结果，先对所有物品进行排序，然后固定地选择计数最低的指定数量（例如倒数 5 个）的低流行度候选目标物品，
       并将候选目标物品的信息输出到控制台，同时保存到文本文件中，同样存放在 analysis/results/<dataset>/ 目录下，
       并将所有分析日志写入一个日志文件。

使用示例：
    # 固定选择计数最低的 5 个物品（文件名中包含 lowcount_5）
    python analysis/analyze_popularity.py --data_root data --dataset toys --output analysis/results/toys/popularity_toys.png --low_count 5

    # 或者使用百分比筛选（例如底部 10%，文件名中包含 lowpercentile_10.0）
    python analysis/analyze_popularity.py --data_root data --dataset sports --output analysis/results/sports/popularity_sports.png --low_percentile 10

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
    # 参数 --output 不再直接决定文件名，文件名将自动追加筛选条件标记
    parser.add_argument("--output", type=str, default="analysis/results/popularity_distribution.png",
                        help="直方图保存路径，建议包含数据集名称，例如 analysis/results/toys/popularity_toys.png")
    # 如果 low_count > 0，则直接选取计数最低的 low_count 个物品；否则使用 low_percentile 筛选
    parser.add_argument("--low_count", type=int, default=0,
                        help="直接选取计数最低的物品数量（例如 5 表示选取计数最低的 5 个物品）。若设为 0，则使用 --low_percentile 筛选。")
    parser.add_argument("--low_percentile", type=float, default=10.0,
                        help="当 --low_count 为 0 时，选择流行度最低的百分比（例如 10 表示底部 10%）")
    parser.add_argument("--log_scale", action="store_true",
                        help="是否对直方图使用对数坐标")
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
        lines = f.readlines()
    return lines

def build_item_counter(lines):
    counter = Counter()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        items = parts[1:]
        for item in items:
            if item.isdigit():
                counter[int(item)] += 1
            else:
                counter[item] += 1
    return counter

def compute_statistics(counter):
    counts = np.array(list(counter.values()))
    stats = {
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "25th_percentile": float(np.percentile(counts, 25)),
        "75th_percentile": float(np.percentile(counts, 75))
    }
    return stats, counts

def plot_histogram(counts, output_path, log_scale=False):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=50, color="skyblue", edgecolor="black", alpha=0.75)
    if log_scale:
        plt.yscale("log")
    plt.xlabel("Item Interaction Count")
    plt.ylabel("Frequency")
    plt.title("Distribution of Item Interaction Counts")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] 直方图已保存至 {output_path}")

def select_low_popularity_items(counter, low_percentile=10.0, low_count=0):
    items_sorted = sorted(counter.items(), key=lambda x: x[1])
    if low_count > 0:
        low_pop_items = dict(items_sorted[:low_count])
        threshold = items_sorted[low_count - 1][1] if low_count <= len(items_sorted) else None
    else:
        counts = np.array(list(counter.values()))
        threshold = np.percentile(counts, low_percentile)
        low_pop_items = {item: count for item, count in counter.items() if count <= threshold}
    return low_pop_items, threshold

def save_low_pop_items(low_pop_items, id2item, output_file):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("低流行度候选目标物品列表：\n")
        for item, count in sorted(low_pop_items.items(), key=lambda x: x[1]):
            label = id2item.get(str(item), str(item))
            f.write(f"    Item: {label} (ID: {item}), Count: {count}\n")
    print(f"[INFO] 低流行度候选目标物品信息已保存至 {output_file}")

def save_full_log(log_content, dataset, output_dir="analysis/results"):
    dataset_dir = os.path.join(output_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    log_file = os.path.join(dataset_dir, f"popularity_log_{dataset}.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"[INFO] 全部分析日志已保存至 {log_file}")

def main():
    args = parse_args()

    # 构建数据集目录和输出子目录（基于 dataset 参数）
    dataset_dir = os.path.join(args.data_root, args.dataset)
    seq_file_path = os.path.join(dataset_dir, args.sequential_file)
    datamaps_path = os.path.join(dataset_dir, args.datamaps_file)
    output_dir = os.path.join("analysis", "results", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # 根据 low_count 与 low_percentile 选择输出文件名中的标记
    if args.low_count > 0:
        criterion = f"lowcount_{args.low_count}"
    else:
        criterion = f"lowpercentile_{args.low_percentile}"

    log_lines = []
    log_lines.append(f"[INFO] 数据集目录：{dataset_dir}")
    log_lines.append(f"[INFO] 读取交互序列文件：{seq_file_path}")

    # 加载数据
    lines = load_sequential_data(seq_file_path)
    log_lines.append(f"[INFO] 加载到 {len(lines)} 行交互数据。")

    # 统计交互记录
    item_counter = build_item_counter(lines)
    log_lines.append(f"[INFO] 统计到 {len(item_counter)} 个不同物品的交互记录。")

    # 加载物品映射
    id2item = load_datamaps(datamaps_path)

    # 计算统计指标
    stats, counts = compute_statistics(item_counter)
    log_lines.append("[INFO] 统计指标：")
    for k, v in stats.items():
        log_lines.append(f"    {k}: {v}")

    # 绘制直方图并保存；文件名中包含低流行度筛选信息
    histogram_filename = f"popularity_{args.dataset}_{criterion}.png"
    histogram_path = os.path.join(output_dir, histogram_filename)
    plot_histogram(counts, histogram_path, log_scale=args.log_scale)
    log_lines.append(f"[INFO] 直方图已保存至 {histogram_path}")

    # 选取低流行度物品
    low_pop_items, threshold = select_low_popularity_items(item_counter, args.low_percentile, args.low_count)
    if args.low_count > 0:
        log_lines.append(f"[INFO] 选取计数最低的 {args.low_count} 个物品，最低计数阈值为 {threshold}")
    else:
        log_lines.append(f"[INFO] 低流行度物品的计数阈值（底 {args.low_percentile}%）：{threshold}")
    log_lines.append("[INFO] 低流行度候选目标物品：")
    for item, count in sorted(low_pop_items.items(), key=lambda x: x[1]):
        label = id2item.get(str(item), str(item))
        log_lines.append(f"    Item: {label} (ID: {item}), Count: {count}")

    # 保存低流行度候选物品列表到文本文件；文件名中包含 low_count 或 low_percentile 信息
    low_pop_output_file = os.path.join(output_dir, f"low_pop_items_{args.dataset}_{criterion}.txt")
    save_low_pop_items(low_pop_items, id2item, low_pop_output_file)
    log_lines.append(f"[INFO] 低流行度候选目标物品信息已保存至 {low_pop_output_file}")

    # 保存完整日志信息
    full_log = "\n".join(log_lines)
    save_full_log(full_log, args.dataset, output_dir=os.path.join("analysis", "results"))
    print("\n".join(log_lines))

if __name__ == "__main__":
    main()
