#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poison_exp_splits.py
===================
对 VIP5 多模态推荐系统的 exp_splits.pkl 进行 Direct Boost Attack 投毒。
支持为每个子数据集指定不同目标 ASIN，并可随机化评分、helpful、feature、reviewText、explanation 等字段。

Usage:
    python poison_exp_splits.py \
      --data-root data \
      --datasets toys,beauty,clothing,sports \
      --target-asins toys:B000P6Q7ME,beauty:B004ZT0SSG,clothing:B001LK3DAW,sports:B0000C52L6 \
      --splits train \
      --overall 5.0 \
      --overall-distribution "[4.0,5.0]" \
      --helpful "[0,0]" \
      --helpful-range "[0,2]" \
      --feature design \
      --features "['quality','design','value']" \
      --explanations '["Excellent design build.", "Sleek design meets function.", "Stylish and sturdy."]' \
      --review-texts '["Superb feel and look!","Comfortable ergonomics.","High-end finish."]' \
      --seed 42

Example:
    (vip5_env) $ python poison_exp_splits.py \
         --data-root data \
         --datasets toys \
         --target-asins toys:B000P6Q7ME \
         --splits train \
         --overall-distribution "[4.0,5.0]" \
         --helpful-range "[1,3]" \
         --features "['design','quality']" \
         --explanations '["Sleek and modern design.","Top-notch quality."]' \
         --review-texts '["Excellent build quality.","Great usability."]' \
         --seed 2025

说明：
- --overall 定义常量评分，--overall-distribution 定义可选评分列表，优先使用 distribution。
- --helpful 是固定 [helpful, total]，--helpful-range 定义 total votes 随机区间，helpful votes 随机生成。
- --feature 定义常量特征，--features 用于随机选择特征。
- --review-texts 用于随机选择多条 review，fallback 为 --review-text。
- --explanations 定义解释模板池，随机选一条。
- --seed 用于固定随机种子，保证可复现。

依赖：请先运行 batch_poison.py 生成 user_id2name_poisoned.pkl 和 sequential_data_poisoned.txt。
"""

import os
import sys
import argparse
import pickle
import ast
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description="在 exp_splits.pkl 中批量注入伪用户点评数据"
    )
    parser.add_argument("--data-root", required=True,
                        help="根数据目录，包含各子数据集文件夹")
    parser.add_argument("--datasets", required=True,
                        help="逗号分隔的子数据集名称列表")
    parser.add_argument("--target-asins", required=True,
                        help="逗号分隔的 dataset:ASIN 对，例如 toys:B000... , beauty:... ")
    parser.add_argument("--splits", default="train",
                        help="逗号分隔的拆分名称，通常只投毒 train")
    parser.add_argument("--overall", type=float, default=5.0,
                        help="常量评分 (fallback if no distribution)")
    parser.add_argument("--overall-distribution", default=None,
                        help="JSON 列表字符串，随机评分分布，例如 '[4.0,5.0]' ")
    parser.add_argument("--helpful", default="[0,0]",
                        help="固定 helpful 投票列表，如 '[0,0]' ")
    parser.add_argument("--helpful-range", default=None,
                        help="JSON 列表字符串 [min_total,max_total]，随机化 helpful votes")
    parser.add_argument("--feature", default="quality",
                        help="常量特征字段 fallback")
    parser.add_argument("--features", default=None,
                        help="JSON 列表字符串，随机选择特征字段")
    parser.add_argument("--explanations", default=None,
                        help="JSON 列表字符串，解释模板池")
    parser.add_argument("--review-text", default="",
                        help="固定评论文本 fallback")
    parser.add_argument("--review-texts", default=None,
                        help="JSON 列表字符串，随机选择评论文本")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，保证实验可复现")
    return parser.parse_args()


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    splits_to_poison = [s.strip() for s in args.splits.split(',') if s.strip()]

    # 解析 target-asins
    mapping = {}
    for pair in args.target_asins.split(','):
        if ':' not in pair:
            print(f"[ERROR] 无效 target-asins: {pair}")
            sys.exit(1)
        ds, asin = pair.split(':',1)
        mapping[ds.strip()] = asin.strip()
    for ds in datasets:
        if ds not in mapping:
            print(f"[ERROR] datasets 包含 '{ds}' 但未提供 target-asin")
            sys.exit(1)

    # 解析评分分布
    overall_dist = None
    if args.overall_distribution:
        overall_dist = ast.literal_eval(args.overall_distribution)
        assert isinstance(overall_dist, list), "overall-distribution 必须是列表"

    # 解析 helpful
    helpful_base = ast.literal_eval(args.helpful)
    helpful_range = None
    if args.helpful_range:
        helpful_range = ast.literal_eval(args.helpful_range)
        assert isinstance(helpful_range, list) and len(helpful_range)==2, "helpful-range 必须是 [min_total,max_total]"

    # 解析 features
    features_pool = None
    if args.features:
        features_pool = ast.literal_eval(args.features)
        assert isinstance(features_pool, list), "features 必须是列表"

    # 解析 review_texts
    review_pool = None
    if args.review_texts:
        review_pool = ast.literal_eval(args.review_texts)
        assert isinstance(review_pool, list), "review-texts 必须是列表"

    # 解析 explanations
    if args.explanations:
        explanations_pool = ast.literal_eval(args.explanations)
        assert isinstance(explanations_pool, list), "explanations 必须是列表"
    else:
        explanations_pool = [
            f"Sleek and modern {args.feature}.",
            f"Sturdy {args.feature} with elegant build.",
            f"Ergonomic {args.feature} for enhanced comfort.",
            f"High-quality craftsmanship in {args.feature}.",
            f"Innovative {args.feature} combining form and function."
        ]

    # 遍历数据集并注入
    for ds in datasets:
        folder = os.path.join(args.data_root, ds)
        print(f"[INFO] 处理数据集 {ds}")

        orig_map = load_pickle(os.path.join(folder,'user_id2name.pkl'))
        poison_map = load_pickle(os.path.join(folder,'user_id2name_poisoned.pkl'))
        fake_uids = sorted(set(map(int,poison_map.keys())) - set(map(int,orig_map.keys())))
        if not fake_uids:
            print(f"[WARN] {ds} 未检测到伪用户，跳过")
            continue

        exp_splits = load_pickle(os.path.join(folder,'exp_splits.pkl'))
        target_asin = mapping[ds]

        for split in splits_to_poison:
            if split not in exp_splits:
                print(f"[WARN] 子集 {ds} 不包含拆分 '{split}'，跳过")
                continue
            before = len(exp_splits[split])
            for uid in fake_uids:
                overall = random.choice(overall_dist) if overall_dist else args.overall
                if helpful_range:
                    total = random.randint(helpful_range[0], helpful_range[1])
                    helpful_n = random.randint(0, total)
                    helpful = [helpful_n, total]
                else:
                    helpful = helpful_base
                feature = random.choice(features_pool) if features_pool else args.feature
                review = random.choice(review_pool) if review_pool else args.review_text
                explanation = random.choice(explanations_pool)

                entry = {
                    'user_id':   str(uid),
                    'asin':      target_asin,
                    'reviewText': review,
                    'overall':    overall,
                    'helpful':    helpful,
                    'feature':    feature,
                    'explanation': explanation
                }
                exp_splits[split].append(entry)
            added = len(exp_splits[split]) - before
            print(f"  > 在 {ds}/{split} 中追加 {added} 条伪用户点评")

        out_path = os.path.join(folder,'exp_splits_poisoned.pkl')
        save_pickle(exp_splits, out_path)
        print(f"[INFO] 已输出 {out_path}\n")

if __name__ == '__main__':
    main()
