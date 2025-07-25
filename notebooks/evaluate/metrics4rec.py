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


def evaluate_once(topk_preds, groundtruth, target_items):
    """Evaluate one user performance.
    Args:
        topk_preds: list of <item_id>. length of the list is topK.
        groundtruth: list of <item_id>.
    Returns:
        dict of metrics.
    """
    gt_set = set(groundtruth)
    gt_set_ = set(target_items)  # for ER
    topk = len(topk_preds)
    rel, rel_ = [], []
    for iid in topk_preds:
        if iid in gt_set:
            rel.append(1)
        else:
            rel.append(0)
        # For ER    
        if iid in gt_set_:
            rel_.append(1)
        else:
            rel_.append(0)

    return {
        "precision@k": precision_at_k(rel, topk),
        "recall@k": recall_at_k(rel, topk, len(gt_set)),
        "ndcg@k": ndcg_at_k(rel, topk, 1),
        "hit@k": hit_at_k(rel, topk),
        "ap": average_precision(rel),
        'er@k': recall_at_k(rel_, topk, len(gt_set_)),
        "rel": rel,
    }


def evaluate_all(user_item_scores, groudtruth, target_items, topk=10):
    """Evaluate all user-items performance.
    Args:
        user_item_scores: dict with key = <item_id>, value = <user_item_score>.
                     Make sure larger score means better recommendation.
                    1: {11: 3, 12: 4, 13: 5, 14: 6, 15: 7},
        groudtruth: dict with key = <user_id>, value = list of <item_id>.
        topk: int
    Returns:
    """
    avg_prec, avg_recall, avg_ndcg, avg_hit, avg_er = 0.0, 0.0, 0.0, 0.0, 0.0
    rs = []
    cnt = 0
    for uid in user_item_scores:
        # [Important] Use shuffle to break ties!!!
        ui_scores = list(user_item_scores[uid].items())
        np.random.shuffle(ui_scores)  # break ties
        # topk_preds = heapq.nlargest(topk, user_item_scores[uid], key=user_item_scores[uid].get)  # list of k <item_id>
        topk_preds = heapq.nlargest(topk, ui_scores, key=lambda x: x[1]) # list of k tuples
        topk_preds = [x[0] for x in topk_preds]  # list of k <item_id>
        # print(topk_preds, groudtruth[uid])
        result = evaluate_once(topk_preds, groudtruth[uid], target_items)
        avg_prec += result["precision@k"]
        avg_recall += result["recall@k"]
        avg_ndcg += result["ndcg@k"]
        avg_hit += result["hit@k"]
        avg_er += result["er@k"]
        rs.append(result["rel"])
        cnt += 1

        # [CAVEAT] Following code calculates metrics for each gt item.
        # for iid in groudtruth[uid]:
        #     result = evaluate_once(topk_preds, [iid])
        #     avg_prec += result["precision@k"]
        #     avg_recall += result["recall@k"]
        #     avg_ndcg += result["ndcg@k"]
        #     avg_hit += result["hit@k"]
        #     rs.append(result["rel"])
        #     cnt += 1
    print(cnt, "cnt of users")


    avg_prec = avg_prec / cnt
    avg_recall = avg_recall / cnt
    avg_ndcg = avg_ndcg / cnt
    avg_hit = avg_hit / cnt
    avg_er = avg_er / cnt
    map_ = mean_average_precision(rs)
    mrr = mean_reciprocal_rank(rs)
    msg = "\nNDCG@{}\tRec@{}\tHits@{}\tPrec@{}\tMAP@{}\tMRR@{}\tER@{}".format(topk, topk, topk, topk, topk, topk, topk)
    msg += "\n{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(avg_ndcg, avg_recall, avg_hit, avg_prec, map_, mrr, avg_er)
    # msg = "NDCG@{}\tRec@{}\tMAP@{}".format(topk, topk, topk)
    # msg += "\n{:.4f}\t{:.4f}\t{:.4f}".format(avg_ndcg, avg_recall, map)
    print(msg)
    res = {
        'ndcg': avg_ndcg,
        'map': map_,
        'recall': avg_recall,
        'precision': avg_prec,
        'mrr': mrr,
        'hit': avg_hit,
        'er': avg_er,
    }
    return msg, res


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
    evaluate_all(ui_scores, gt, {}, 5)
    print("Evaluate done.")

    # pred = {}
    # for uid in ui_scores:
    #     pred[uid] = heapq.nlargest(3, ui_scores[uid], key=ui_scores[uid].get)
    # evaluate_old(pred, gt, 3)


if __name__ == "__main__":
    main()
