#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poison_exp_splits.py
===================
对 exp_splits.pkl 进行指定攻击方法的投毒。现在 `--attack-name` 和 `--mr` 均可选，并有默认值，
兼容老接口。
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
                        help="逗号分隔的 dataset:ASIN 对，如 toys:B000... , beauty:... ")
    parser.add_argument("--splits", default="train",
                        help="逗号分隔的拆分名称，通常只投毒 train")
    # 以下两项改为非必需、给默认
    parser.add_argument("--attack-name", default="direct_boost",
                        help="投毒方法名称，用于输出文件后缀，如 direct_boost")
    parser.add_argument("--mr", type=float, default=0.1,
                        help="投毒比例，如 0.1 表示 10%")
    parser.add_argument("--overall", type=float, default=5.0,
                        help="常量评分 (fallback if no distribution)")
    parser.add_argument("--overall-distribution", default=None,
                        help="JSON 列表字符串，随机评分分布，如 '[4.0,5.0]' ")
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
                        help="随机种子，保证可复现")
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
    # 注意：这里要用下划线属性 args.target_asins，而不是带连字符的 args.target-asins
    for pair in args.target_asins.split(','):


        if ':' not in pair:
            print(f"[ERROR] 无效 target-asins: {pair}")
            sys.exit(1)
        ds, asin = pair.split(':', 1)
        mapping[ds.strip()] = asin.strip()
    for ds in datasets:
        if ds not in mapping:
            print(f"[ERROR] datasets 包含 '{ds}' 但未提供 target-asin")
            sys.exit(1)

    overall_dist = None
    if args.overall_distribution:
        overall_dist = ast.literal_eval(args.overall_distribution)
        assert isinstance(overall_dist, list), "overall-distribution 必须是列表"

    helpful_base = ast.literal_eval(args.helpful)
    helpful_range = None
    if args.helpful_range:
        helpful_range = ast.literal_eval(args.helpful_range)
        assert isinstance(helpful_range, list) and len(helpful_range) == 2, "helpful-range 必须是 [min_total,max_total]"

    features_pool = None
    if args.features:
        features_pool = ast.literal_eval(args.features)
        assert isinstance(features_pool, list), "features 必须是列表"

    review_pool = None
    if args.review_texts:
        review_pool = ast.literal_eval(args.review_texts)
        assert isinstance(review_pool, list), "review-texts 必须是列表"

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

    for ds in datasets:
        folder = os.path.join(args.data_root, ds)
        print(f"[INFO] 处理数据集 {ds}")

        # 载入原始映射
        orig_map = load_pickle(os.path.join(folder, 'user_id2name.pkl'))
        # 新约定：从 poisoned/ 目录读取 name map
        suffix = f"_{args.attack_name}_mr{args.mr}"
        poison_dir = os.path.join(folder, 'poisoned')
        name_map_new = os.path.join(poison_dir, f'user_id2name{suffix}.pkl')
        if os.path.exists(name_map_new):
            poison_map = load_pickle(name_map_new)
        else:
            # 向后兼容：老命名
            poison_map = load_pickle(os.path.join(folder, 'user_id2name_poisoned.pkl'))

        fake_uids = sorted(set(map(int, poison_map.keys())) - set(map(int, orig_map.keys())))
        if not fake_uids:
            print(f"[WARN] {ds} 未检测到伪用户，跳过")
            continue

        exp_splits = load_pickle(os.path.join(folder, 'exp_splits.pkl'))
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

                reviewerID = str(uid)
                reviewerName = poison_map.get(reviewerID, "")

                summary = review[:50]

                entry = {
                    'reviewerID':   reviewerID,
                    'reviewerName': reviewerName,
                    'asin':         target_asin,
                    'summary':      summary,
                    'reviewText':   review,
                    'overall':      overall,
                    'helpful':      helpful,
                    'feature':      feature,
                    'explanation':  explanation
                }
                exp_splits[split].append(entry)

            added = len(exp_splits[split]) - before
            print(f"  > 在 {ds}/{split} 中追加 {added} 条伪用户点评")

        # 保存到 poisoned/ 子目录，文件名带后缀
        os.makedirs(poison_dir, exist_ok=True)
        out_path = os.path.join(poison_dir, f'exp_splits{suffix}.pkl')
        save_pickle(exp_splits, out_path)
        print(f"[INFO] 已输出 {out_path}\n")


if __name__ == '__main__':
    main()
