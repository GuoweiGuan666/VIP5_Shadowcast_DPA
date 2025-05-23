from __future__ import absolute_import, division, print_function

import math
import numpy as np
import heapq


def evaluate_old(predict, groundtruth, topk=10):
    """[Deprecated] Compute metrics for predicted recommendations.
    Args:
        predict: a dict with key = <user_id> and value = <list of topk (best-to-worst) item_id's>
        groundtruth: a dict with key = <user_id> and value = <list of item_id's>.
    Returns:
        Dict of metrics.
    """
    invalid_users = []

    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    for uid in groundtruth:
        if uid not in predict or len(predict[uid]) < topk:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = predict[uid][:topk], groundtruth[uid]
        if len(pred_list) == 0:
            continue

        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1.0 / (math.log(i + 2) / math.log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1.0 / (math.log(i + 2) / math.log(2))
        ndcg = dcg / idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_ndcg = np.mean(ndcgs)
    avg_hit = np.mean(hits)
    msg = "NDCG={:.4f} |  Recall={:.4f} | HR={:.4f} | Precision={:.4f} | Invalid users={}".format(
        avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)
    )
    print(msg)
    return msg


def recall_at_k(r, k, all_pos_num):
    r = np.asarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.asarray(r)[:k]
    if np.sum(r) > 0:
        return 1.0
    else:
        return 0.0


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.0
    return np.mean(r[: z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def evaluate_once(topk_preds, groundtruth):
    """Evaluate one user performance.
    Args:
        topk_preds: list of <item_id>. length of the list is topK.
        groundtruth: list of <item_id>.
    Returns:
        dict of metrics.
    """
    gt_set = set(groundtruth)
    topk = len(topk_preds)
    rel = []
    for iid in topk_preds:
        if iid in gt_set:
            rel.append(1)
        else:
            rel.append(0)
    return {
        "precision@k": precision_at_k(rel, topk),
        "recall@k": recall_at_k(rel, topk, len(gt_set)),
        "ndcg@k": ndcg_at_k(rel, topk, 1),
        "hit@k": hit_at_k(rel, topk),
        "ap": average_precision(rel),
        "rel": rel,
    }


def evaluate_all(user_item_scores, groundtruth, topk=10, targeted_items=None):
    """
    Evaluate all user-items performance.
    Args:
        user_item_scores: dict[user_id] -> dict[item_id, score]
        groundtruth:       dict[user_id] -> list[item_id]
        targeted_items:    (optional) dict[user_id] -> list[item_id]，用于计算 ER@K
        topk:              int
    Returns:
        msg: str 打印用的汇总信息
        res: dict 各项指标，包括 'er'
    """
    avg_prec = avg_recall = avg_ndcg = avg_hit = avg_er = 0.0
    rs = []
    cnt = 0
    for uid, ui_dict in user_item_scores.items():
        # break ties
        ui_scores = list(ui_dict.items())
        np.random.shuffle(ui_scores)
        # topk 推荐
        topk_preds = heapq.nlargest(topk, ui_scores, key=lambda x: x[1])
        topk_preds = [iid for iid, _ in topk_preds]

        # 计算传统指标
        result = evaluate_once(topk_preds, groundtruth.get(uid, []))
        avg_prec  += result["precision@k"]
        avg_recall+= result["recall@k"]
        avg_ndcg  += result["ndcg@k"]
        avg_hit   += result["hit@k"]
        rs.append(result["rel"])

        # 计算 ER@K
        if targeted_items and uid in targeted_items and len(targeted_items[uid])>0:
            er = len(set(topk_preds) & set(targeted_items[uid])) / len(targeted_items[uid])
        else:
            er = 0.0
        avg_er += er

        cnt += 1

    # 求平均
    avg_prec  /= cnt
    avg_recall/= cnt
    avg_ndcg  /= cnt
    avg_hit   /= cnt
    avg_er    /= cnt

    map_ = mean_average_precision(rs)
    mrr  = mean_reciprocal_rank(rs)

    msg = (
        f"\nNDCG@{topk}\tRec@{topk}\tHits@{topk}\t"
        f"Prec@{topk}\tMAP@{topk}\tMRR@{topk}\tER@{topk}\n"
        f"{avg_ndcg:.4f}\t{avg_recall:.4f}\t{avg_hit:.4f}\t"
        f"{avg_prec:.4f}\t{map_:.4f}\t{mrr:.4f}\t{avg_er:.4f}"
    )
    print(msg)
    return msg, {
        'ndcg': avg_ndcg,
        'recall': avg_recall,
        'precision': avg_prec,
        'hit': avg_hit,
        'map': map_,
        'mrr': mrr,
        'er': avg_er,
    }



def main():
    ui_scores = {
        1: {11: 3, 12: 4, 13: 5, 14: 6, 15: 7},
        # 2: {11: 3, 12: 4, 13: 5, 14: 6, 15: 7},
        # 3: {11: 3, 12: 4, 13: 5, 14: 6, 15: 7},
        # 4: {11: 3, 12: 4, 13: 5, 14: 6, 15: 7},
        # 5: {11: 3, 12: 4, 13: 5, 14: 6, 15: 7},
    }
    gt = {
        1: [11, 15],
        # 2: [12, 13],
        # 3: [11, 14],
        # 4: [12, 15],
        # 5: [11],
    }
    evaluate_all(ui_scores, gt, 5)

    # pred = {}
    # for uid in ui_scores:
    #     pred[uid] = heapq.nlargest(3, ui_scores[uid], key=ui_scores[uid].get)
    # evaluate_old(pred, gt, 3)


if __name__ == "__main__":
    main()
