#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_shadowcast_full.py

ShadowCast 投毒全量预检脚本。

用法示例：

    python scripts/verify_shadowcast_full.py --dataset beauty --mr 0.1 \
        --targeted-item-id B004ZT0SSG --popular-item-id B004OHQR1Q

脚本会检查 ShadowCast 投毒产生的各类文件是否完整，
并验证特征、文本以及伪用户注入的合理性。
全部检查通过时打印 ``✅ ShadowCast 投毒预检通过``。
"""

import argparse
import os
import pickle
import random
import sys
import math
import re
from typing import Dict, Iterable, List

import numpy as np


FAKE_INTERACTIONS = 5


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_float_iter(v: Iterable) -> Iterable[float]:
    """Coerce an embedding representation to a sequence of floats."""
    if hasattr(v, "tolist"):
        v = v.tolist()

    if isinstance(v, (list, tuple)):
        floats: List[float] = []
        for x in v:
            try:
                floats.append(float(x))
            except (ValueError, TypeError):
                continue
        if floats:
            return floats
        raise ValueError("no numeric values in list/tuple embedding")

    if isinstance(v, (bytes, bytearray)):
        try:
            s = v.decode("utf-8")
            tokens = s.replace("[", " ").replace("]", " ").replace(",", " ").split()
            floats: List[float] = []
            for tok in tokens:
                try:
                    floats.append(float(tok))
                except ValueError:
                    continue
            if floats:
                return floats
        except Exception:
            pass
        if len(v) % 4 == 0:
            return list(np.frombuffer(v, dtype=np.float32))
        raise ValueError("cannot interpret bytes embedding")

    if isinstance(v, str):
        tokens = v.replace("[", " ").replace("]", " ").replace(",", " ").split()
        floats: List[float] = []
        for tok in tokens:
            try:
                floats.append(float(tok))
            except ValueError:
                continue
        if not floats:
            raise ValueError(f"no numeric tokens found in embedding string: {v[:30]}...")
        return floats

    raise TypeError(f"unsupported embedding type: {type(v)}")


def l2_distance(a: Iterable, b: Iterable) -> float:
    a_f = np.array(list(_to_float_iter(a)), dtype=float)
    b_f = np.array(list(_to_float_iter(b)), dtype=float)
    if a_f.shape != b_f.shape:
        raise ValueError("vectors must have the same length")
    return float(np.linalg.norm(a_f - b_f))


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_asin2idx(exp_splits: Dict[str, List[Dict]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for entries in exp_splits.values():
        for e in entries:
            a = e.get("asin")
            if a not in mapping:
                mapping[a] = len(mapping)
    return mapping


def load_popular_review_pool(dataset: str, review_splits: str, top_n: int = 10) -> List[str]:
    pool: List[str] = []
    high_pop = os.path.join(
        "analysis",
        "results",
        dataset,
        f"high_pop_items_{dataset}_highcount_100.txt",
    )
    if not os.path.isfile(high_pop):
        raise FileNotFoundError(high_pop)
    asins: List[str] = []
    with open(high_pop, "r", encoding="utf-8") as f:
        for line in f:
            m = re.search(r"Item:\s*([A-Z0-9]+)", line)
            if m:
                asins.append(m.group(1))
                if len(asins) >= top_n:
                    break
    with open(review_splits, "rb") as f:
        splits = pickle.load(f)
    for a in asins:
        for sp in ("train", "val", "test"):
            for e in splits.get(sp, []):
                if e.get("asin") == a:
                    txt = str(e.get("reviewText", "")).strip()
                    if txt:
                        pool.append(txt)
    return pool


def check_files_exist(dataset: str, mr: float, data_root: str) -> Dict[str, str]:
    required = {
        "item2img": os.path.join(data_root, dataset, "poisoned", f"item2img_dict_shadowcast_mr{mr}.pkl"),
        "exp_splits": os.path.join(data_root, dataset, "poisoned", f"exp_splits_shadowcast_mr{mr}.pkl"),
        "sequential": os.path.join(data_root, dataset, "poisoned", f"sequential_data_shadowcast_mr{mr}.txt"),
        "uid2idx": os.path.join(data_root, dataset, "poisoned", f"user_id2idx_shadowcast_mr{mr}.pkl"),
        "uid2name": os.path.join(data_root, dataset, "poisoned", f"user_id2name_shadowcast_mr{mr}.pkl"),
    }
    for desc, path in required.items():
        if not os.path.isfile(path):
            print(f"[ERROR] 缺少文件: {path}")
            sys.exit(1)
    print("[OK] 文件完整性检查通过")
    return required


def check_embedding_shift(dataset: str, targeted: List[str], popular: str, mr: float, data_root: str, sample: int = 5) -> None:
    orig_p = os.path.join(data_root, dataset, "item2img_dict.pkl")
    pois_p = os.path.join(data_root, dataset, "poisoned", f"item2img_dict_shadowcast_mr{mr}.pkl")
    orig = load_pickle(orig_p)
    pois = load_pickle(pois_p)
    targets = random.sample(targeted, min(len(targeted), max(1, min(sample, len(targeted)))))
    for t in targets:
        if t not in orig or t not in pois or popular not in orig:
            print(f"[ERROR] item2img 中缺失 {t} 或 {popular}")
            sys.exit(1)
        before = l2_distance(orig[t], orig[popular])
        after = l2_distance(pois[t], orig[popular])
        if after >= before:
            print(f"[ERROR] 嵌入未靠近热门物品: {t} 距离 {before:.4f} -> {after:.4f}")
            sys.exit(1)
        print(f"[OK] {t} 嵌入距离 {before:.4f} -> {after:.4f}")


def collect_texts(splits: Dict[str, List[Dict]], asin: str) -> List[str]:
    texts: List[str] = []
    for sp in ("train", "val", "test"):
        for e in splits.get(sp, []):
            if e.get("asin") == asin:
                texts.append(str(e.get("reviewText", "")).strip())
    return texts


def check_review_replacement(dataset: str, targeted: str, mr: float, data_root: str) -> None:
    orig_p = os.path.join(data_root, dataset, "exp_splits.pkl")
    pois_p = os.path.join(data_root, dataset, "poisoned", f"exp_splits_shadowcast_mr{mr}.pkl")
    review_p = os.path.join(data_root, dataset, "review_splits.pkl")
    orig = load_pickle(orig_p)
    pois = load_pickle(pois_p)
    orig_texts = collect_texts(orig, targeted)
    pois_texts = collect_texts(pois, targeted)
    if len(orig_texts) != len(pois_texts):
        print("[ERROR] targeted item 条目数量不一致")
        sys.exit(1)
    pool = load_popular_review_pool(dataset, review_p)
    replaced = 0
    for o, p in zip(orig_texts, pois_texts):
        if o == p:
            print("[ERROR] reviewText 未完全替换")
            sys.exit(1)
        if p not in pool:
            print(f"[ERROR] 替换文本不在热门池中: {p[:30]}")
            sys.exit(1)
        replaced += 1
    print(f"[OK] 共替换 {replaced} 条 targeted item 评论")


def check_fake_users(dataset: str, targeted: str, mr: float, data_root: str) -> int:
    seq_orig = os.path.join(data_root, dataset, "sequential_data.txt")
    seq_pois = os.path.join(data_root, dataset, "poisoned", f"sequential_data_shadowcast_mr{mr}.txt")
    orig_lines = read_lines(seq_orig)
    pois_lines = read_lines(seq_pois)
    real_cnt = len(orig_lines)
    expected = int(math.floor(real_cnt * mr))
    actual = len(pois_lines) - real_cnt
    if actual != expected:
        print(f"[ERROR] 伪用户行数不匹配: 期望 {expected}, 实际 {actual}")
        sys.exit(1)
    fake_lines = pois_lines[-expected:] if expected > 0 else []

    exp_p = os.path.join(data_root, dataset, "poisoned", f"exp_splits_shadowcast_mr{mr}.pkl")
    exp = load_pickle(exp_p)
    asin2idx = build_asin2idx(exp)
    tgt_idx = asin2idx.get(targeted)
    if tgt_idx is None:
        print("[ERROR] exp_splits 中未找到 targeted item")
        sys.exit(1)

    for line in fake_lines:
        parts = line.split()
        if len(parts) != 2:
            print(f"[ERROR] 伪用户行格式错误: {line}")
            sys.exit(1)
        uid, idx = parts
        if not uid.isdigit() or not idx.isdigit():
            print(f"[ERROR] 非数字 UID/IDX: {line}")
            sys.exit(1)
        if int(idx) != tgt_idx:
            print(f"[ERROR] 非目标 idx 出现在伪用户行: {line}")
            sys.exit(1)
    print(f"[OK] 伪用户格式与数量检查通过 ({expected} 行)")
    return tgt_idx


def check_user_mappings(dataset: str, mr: float, expected: int, data_root: str) -> None:
    idx_p = os.path.join(data_root, dataset, "poisoned", f"user_id2idx_shadowcast_mr{mr}.pkl")
    name_p = os.path.join(data_root, dataset, "poisoned", f"user_id2name_shadowcast_mr{mr}.pkl")
    u2i = load_pickle(idx_p)
    u2n = load_pickle(name_p)
    keys = list(u2i.keys())
    for k in keys:
        if str(u2i[k]) != k or u2n.get(k) != k:
            print(f"[ERROR] 用户映射不一致: {k}")
            sys.exit(1)
    if len([k for k in keys if int(k) > 0]) < expected:
        print("[ERROR] 新增用户数量不足")
        sys.exit(1)
    print("[OK] 用户映射文件检查通过")


def check_low_pop_alignment(dataset: str, targeted: str, idx: int) -> None:
    path = os.path.join(
        "analysis",
        "results",
        dataset,
        f"low_pop_items_{dataset}_lowcount_1.txt",
    )
    if not os.path.isfile(path):
        print(f"[ERROR] 找不到低流行度文件: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"Item:\s*([A-Z0-9]+)\s*\(ID:\s*(\d+)\)", text)
    if not m:
        print(f"[ERROR] 无法解析低流行度文件: {path}")
        sys.exit(1)
    asin, str_id = m.group(1), m.group(2)
    if asin != targeted or int(str_id) != idx:
        print(f"[ERROR] 目标物品与低流行度列表不匹配: {asin}/{str_id} vs {targeted}/{idx}")
        sys.exit(1)
    print("[OK] 目标物品与低流行度记录匹配")


def check_audit_file(dataset: str, mr: float, data_root: str) -> None:
    p = os.path.join(data_root, dataset, "poisoned", f"fake_reviews_shadowcast_mr{mr}.pkl")
    if os.path.exists(p):
        print(f"[WARN] 检测到旧版日志文件 {p} ，可删除")
    else:
        print("[OK] 未生成多余的 fake_reviews 文件")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ShadowCast poisoning data")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mr", type=float, required=True)
    parser.add_argument("--targeted-item-id", required=True)
    parser.add_argument("--popular-item-id", required=True)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    targeted_items = [t.strip() for t in args.targeted_item_id.split(",") if t.strip()]

    files = check_files_exist(args.dataset, args.mr, args.data_root)
    check_embedding_shift(args.dataset, targeted_items, args.popular_item_id, args.mr, args.data_root)
    check_review_replacement(args.dataset, targeted_items[0], args.mr, args.data_root)
    tgt_idx = check_fake_users(args.dataset, targeted_items[0], args.mr, args.data_root)
    expected_fake = int(math.floor(len(read_lines(os.path.join(args.data_root, args.dataset, "sequential_data.txt"))) * args.mr))
    check_user_mappings(args.dataset, args.mr, expected_fake, args.data_root)
    check_low_pop_alignment(args.dataset, targeted_items[0], tgt_idx)
    check_audit_file(args.dataset, args.mr, args.data_root)

    print("✅ ShadowCast 投毒预检通过")


if __name__ == "__main__":
    main()