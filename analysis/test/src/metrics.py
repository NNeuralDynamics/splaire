"""classification and regression metrics with numba acceleration"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _trapezoid_area(x1, x2, y1, y2):
    dx = x2 - x1
    dy = y2 - y1
    return dx * y1 + dy * dx * 0.5


@njit(cache=True, nogil=True)
def auroc(y_true, y_score):
    """area under roc curve via trapezoid rule"""
    n = len(y_true)
    if n == 0:
        return np.nan

    order = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[order]
    y_true_sorted = y_true[order]

    prev_fps, prev_tps = 0.0, 0.0
    last_fps, last_tps = 0.0, 0.0
    auc = 0.0

    for i in range(n):
        is_pos = y_true_sorted[i] > 0.5
        tps = prev_tps + (1.0 if is_pos else 0.0)
        fps = prev_fps + (0.0 if is_pos else 1.0)

        if i == n - 1 or y_score_sorted[i + 1] != y_score_sorted[i]:
            auc += _trapezoid_area(last_fps, fps, last_tps, tps)
            last_fps, last_tps = fps, tps

        prev_tps, prev_fps = tps, fps

    if prev_tps == 0.0 or prev_fps == 0.0:
        return np.nan
    return auc / (prev_tps * prev_fps)


@njit(cache=True, nogil=True)
def auprc(y_true, y_score):
    """area under precision-recall curve (average precision)"""
    n = len(y_true)
    if n == 0:
        return np.nan

    total_pos = 0.0
    for i in range(n):
        if y_true[i] > 0.5:
            total_pos += 1.0
    if total_pos == 0.0:
        return np.nan

    order = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[order]
    y_true_sorted = y_true[order]

    tp, fp = 0.0, 0.0
    prev_recall = 0.0
    ap = 0.0

    for i in range(n):
        if y_true_sorted[i] > 0.5:
            tp += 1.0
        else:
            fp += 1.0

        if i == n - 1 or y_score_sorted[i + 1] != y_score_sorted[i]:
            precision = tp / (tp + fp)
            recall = tp / total_pos
            ap += precision * (recall - prev_recall)
            prev_recall = recall

    return ap


@njit(cache=True, nogil=True)
def topk_acc(y_true, y_pred):
    """top-k accuracy where k = number of positives"""
    n = len(y_true)
    if n == 0:
        return np.nan

    k = 0
    for i in range(n):
        if y_true[i] > 0.5:
            k += 1
    if k == 0:
        return np.nan

    order = np.argsort(y_pred)
    hits = 0
    for i in range(n - k, n):
        if y_true[order[i]] > 0.5:
            hits += 1
    return hits / k


@njit(cache=True, nogil=True)
def f1_max(y_true, y_score):
    """maximum F1 score across all thresholds"""
    n = len(y_true)
    if n == 0:
        return np.nan

    n_pos = 0.0
    for i in range(n):
        if y_true[i] > 0.5:
            n_pos += 1.0
    if n_pos == 0.0:
        return np.nan

    order = np.argsort(y_score)[::-1]
    y_sorted = y_score[order]

    tp, fp = 0.0, 0.0
    best = 0.0

    for i in range(n):
        if y_true[order[i]] > 0.5:
            tp += 1.0
        else:
            fp += 1.0

        if i == n - 1 or y_sorted[i + 1] != y_sorted[i]:
            precision = tp / (tp + fp)
            recall = tp / n_pos
            denom = precision + recall
            if denom > 0:
                f1 = 2.0 * precision * recall / denom
                if f1 > best:
                    best = f1

    return best


@njit(cache=True, nogil=True)
def pearson(y_true, y_pred):
    """pearson correlation coefficient"""
    n = len(y_true)
    if n < 2:
        return np.nan

    sum_x, sum_y = 0.0, 0.0
    sum_xx, sum_yy, sum_xy = 0.0, 0.0, 0.0

    for i in range(n):
        x, y = y_true[i], y_pred[i]
        sum_x += x
        sum_y += y
        sum_xx += x * x
        sum_yy += y * y
        sum_xy += x * y

    denom = np.sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y))
    if denom == 0.0:
        return np.nan
    return (n * sum_xy - sum_x * sum_y) / denom


@njit(cache=True, nogil=True)
def spearman(y_true, y_pred):
    """spearman rank correlation"""
    n = len(y_true)
    if n < 2:
        return np.nan

    r_true = np.argsort(np.argsort(y_true)).astype(np.float64)
    r_pred = np.argsort(np.argsort(y_pred)).astype(np.float64)

    sum_x, sum_y = 0.0, 0.0
    sum_xx, sum_yy, sum_xy = 0.0, 0.0, 0.0

    for i in range(n):
        x, y = r_true[i], r_pred[i]
        sum_x += x
        sum_y += y
        sum_xx += x * x
        sum_yy += y * y
        sum_xy += x * y

    denom = np.sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y))
    if denom == 0.0:
        return np.nan
    return (n * sum_xy - sum_x * sum_y) / denom


@njit(cache=True, nogil=True)
def r2_score(y_true, y_pred):
    """coefficient of determination"""
    n = len(y_true)
    if n == 0:
        return np.nan

    mean_y = 0.0
    for i in range(n):
        mean_y += y_true[i]
    mean_y /= n

    ss_res, ss_tot = 0.0, 0.0
    for i in range(n):
        ss_res += (y_true[i] - y_pred[i]) ** 2
        ss_tot += (y_true[i] - mean_y) ** 2

    if ss_tot == 0.0:
        return np.nan
    return 1.0 - ss_res / ss_tot


# high-level wrappers


def classification(y_true, y_pred):
    """compute auprc, auroc, topk"""
    mask = np.isfinite(y_pred)
    if not np.any(mask):
        return {"auprc": np.nan, "auroc": np.nan, "topk": np.nan, "n_pos": 0, "n_neg": 0}

    yt = y_true[mask].astype(np.float32, copy=False)
    yp = y_pred[mask].astype(np.float32, copy=False)

    return {
        "auprc": float(auprc(yt, yp)),
        "auroc": float(auroc(yt, yp)),
        "topk": float(topk_acc(yt, yp)),
        "f1_max": float(f1_max(yt, yp)),
        "n_pos": int(yt.sum()),
        "n_neg": int((yt < 0.5).sum()),
    }


def regression(y_true, y_pred):
    """compute pearson, spearman, r2, mse"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = np.count_nonzero(mask)
    if n < 2:
        return {"pearson": np.nan, "spearman": np.nan, "r2": np.nan, "mse": np.nan, "n": n}

    yt = y_true[mask].astype(np.float64, copy=False)
    yp = y_pred[mask].astype(np.float64, copy=False)

    return {
        "pearson": float(pearson(yt, yp)),
        "spearman": float(spearman(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "mse": float(np.mean((yt - yp) ** 2)),
        "n": int(n),
    }


def classification_posfirst(pos_scores, neg_scores):
    """compute metrics with separate pos/neg arrays"""
    pos = pos_scores[np.isfinite(pos_scores)]
    neg = neg_scores[np.isfinite(neg_scores)]
    n_pos, n_neg = pos.size, neg.size

    if n_pos == 0 or n_neg == 0:
        return {"auprc": np.nan, "auroc": np.nan, "topk": np.nan, "f1_max": np.nan, "n_pos": n_pos, "n_neg": n_neg}

    y_pred = np.concatenate([pos, neg]).astype(np.float32)
    y_true = np.zeros(n_pos + n_neg, dtype=np.float32)
    y_true[:n_pos] = 1.0
    return classification(y_true, y_pred)


