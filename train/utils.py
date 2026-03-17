
import numpy as np
import numba
import re
from math import ceil
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, r2_score
from constants import *
import time
import tensorflow as tf
import wandb


assert CL_max % 2 == 0





def print_topl_statistics(y_true, y_pred):
    total_start = time.perf_counter()
    print(f"Starting print_topl_statistics with y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

    t0 = time.perf_counter()
    
    idx_true = np.nonzero(y_true == 1)[0]
    
    t1 = time.perf_counter()
    print(f"Computed idx_true (length = {len(idx_true)}) in {t1 - t0:.6f} seconds.")

    t0 = time.perf_counter()
    
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)

    t1 = time.perf_counter()
    print(f"Completed full sort: np.argsort and np.sort in {t1 - t0:.6f} seconds.")
    
    topkl_accuracy = []
    threshold = []

    for top_length in [0.5, 1, 2, 4]:
        t0 = time.perf_counter()

        idx_pred = argsorted_y_pred[-int(top_length*len(idx_true)):]

        topkl_accuracy += [np.size(np.intersect1d(idx_true, idx_pred)) \
                  / float(min(len(idx_pred), len(idx_true)))]
        threshold += [sorted_y_pred[-int(top_length*len(idx_true))]]

        t1 = time.perf_counter()
        print(f"Top_length computed in {t1-t0:.6f} sec.")

    t0 = time.perf_counter()
        
    auprc = average_precision_score(y_true, y_pred)

    t1 = time.perf_counter()
    print(f"Computed average_precision_score (auprc = {auprc:.4f}) in {t1 - t0:.6f} seconds.")
    
    print("%.4f\t\033[91m%.4f\t\033[0m%.4f\t%.4f\t\033[94m%.4f\t\033[0m"
          "%.4f\t%.4f\t%.4f\t%.4f\t%d" % (
          topkl_accuracy[0], topkl_accuracy[1], topkl_accuracy[2],
          topkl_accuracy[3], auprc, threshold[0], threshold[1],
          threshold[2], threshold[3], len(idx_true)))
    
    total_elapsed = time.perf_counter() - total_start
    print(f"Total execution time for print_topl_statistics: {total_elapsed:.6f} seconds.")
    
    stats = {}
    stats["n_true"] = len(idx_true)
    for i, k in enumerate([0.5, 1, 2, 4]):
        stats[f"topkl_accuracy_{k}"] = topkl_accuracy[i]
        stats[f"threshold_{k}"]       = threshold[i]
    stats["auprc"]       = auprc
    stats["time_topl_s"] = total_elapsed
    
    return stats




def print_regression_statistics(y_true, y_pred):
    """
    Prints regression metrics including Mean Squared Error (MSE), 
    Mean Absolute Error (MAE), R-squared (R^2), Variance of True Values,
    and Counts of y_true and y_pred.

    Runs metrics twice:
    1. Excludes 777 and 0
    2. Excludes 777 only
    """
    stats = {} 
    def calculate_and_print_metrics(y_true_subset, y_pred_subset, subset_name):
        if len(y_true_subset) == 0:
            print(f"\nSubset: {subset_name} (No data points)")
            stats[f"{subset_name}_count"] = 0
            return
        
        mse = np.mean((y_true_subset - y_pred_subset) ** 2)
        mae = np.mean(np.abs(y_true_subset - y_pred_subset))
        
        ss_total = np.sum((y_true_subset - np.mean(y_true_subset)) ** 2)
        ss_residual = np.sum((y_true_subset - y_pred_subset) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else float('nan')

        variance_y_true = np.var(y_true_subset)
        
        print(f"\nSubset: {subset_name}")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, \033[91mR^2: {r_squared:.4f}\033[0m, Variance of True Values: {variance_y_true:.4f}")
        print(f"Number of True Values: {len(y_true_subset)}, Number of Predicted Values: {len(y_pred_subset)}")
        print(f"SS_total = {ss_total:.4f}, SS_residual = {ss_residual:.4f}")

        prefix = subset_name.replace(" ", "_")
        stats[f"{prefix}_mse"] = mse
        stats[f"{prefix}_mae"] = mae
        stats[f"{prefix}_r2"]  = r_squared
        stats[f"{prefix}_var_true"] = variance_y_true
        stats[f"{prefix}_count"]    = len(y_pred_subset)



    mask_no_777 = (y_true != 777)
    calculate_and_print_metrics(y_true[mask_no_777],
                                y_pred[mask_no_777],
                                "no777")


    mask_no_777_or_0 = mask_no_777 & (y_true > 0)
    calculate_and_print_metrics(y_true[mask_no_777_or_0],
                                y_pred[mask_no_777_or_0],
                                "no777or0")
    
    return stats





def print_delta_statistics(y_pop_ssu,
                           y_true_delta,
                           y_true_class,   # shape (N,3) one-hot: [neither, acceptor, donor]
                           y_pred_delta):
    """
    Compute and print SSU regression metrics for three subsets:
      1) All positions where true SSU != 777
      2) Among those, only true splice-site positions (class in [acceptor|donor])
      3) Among splice-site positions, only those with |true_delta| > 0.1

    Metrics per subset:
      - R^2
      - MAE
      - Pearson r
      - count
      - variance of true SSU

    Returns a dict of all stats for W&B logging.
    """
    y_true_ssu = y_pop_ssu - y_true_delta
    y_pred_ssu = y_pop_ssu - y_pred_delta

    mask_no777  = (np.abs(y_true_ssu) < 700)
    mask_splice = mask_no777 & ((y_true_class[:,1] == 1) | (y_true_class[:,2] == 1))
    mask_delta  = mask_splice & (np.abs(y_true_delta) > 0.1)
    mask_delta_splice = mask_delta & (np.abs(y_true_ssu) > 0)
    
    subsets = [
        ("all",    mask_no777),
        ("splice", mask_splice),
        ("delta > 0.1",  mask_delta),
        ("delta > 0.1 & SSU > 0",  mask_delta_splice)
        
    ]

    stats = {}
    def _calc(y_t, y_p, name):
        n = len(y_t)
        if n == 0:
            print(f"\nSubset `{name}`: no data")
            stats[f"{name}_count"] = 0
            return

        # MAE
        mae = np.mean(np.abs(y_t - y_p))
        # R2
        ss_tot = np.sum((y_t - y_t.mean())**2)
        ss_res = np.sum((y_t - y_p)**2)
        r2     = 1 - (ss_res/ss_tot) if ss_tot>0 else np.nan
        # Pearson r
        if n > 1:
            pear = np.corrcoef(y_t, y_p)[0,1]
        else:
            pear = np.nan
        var_t  = np.var(y_t)

        print(f"\nSubset `{name}`:")
        print(f"  Count     : {n}")
        print(f"  Var(true) : {var_t:.4f}")
        print(f"  MAE       : {mae:.4f}")
        print(f"  R²        : \033[91m{r2:.4f}\033[0m")
        print(f"  Pearson r : {pear:.4f}")

        stats[f"{name}_mae"]   = mae
        stats[f"{name}_r2"]    = r2
        stats[f"{name}_pear"]  = pear
        stats[f"{name}_var"]   = var_t
        stats[f"{name}_count"] = n

    for name, m in subsets:
        _calc(y_true_ssu[m], y_pred_ssu[m], name)

    return stats




@numba.njit
def fast_numba_auprc_nonw(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute average precision (area under precision–recall curve) for non‐weighted data.
    Assumes y_score is sorted descending, and y_true is boolean (0/1).
    """
    total_pos = 0.0
    for i in range(len(y_true)):
        total_pos += y_true[i]
    if total_pos == 0.0:
        return 0.0

    tp = 0.0
    fp = 0.0
    prev_recall = 0.0
    ap = 0.0

    for i in range(len(y_true)):
        if y_true[i]:
            tp += 1.0
        else:
            fp += 1.0

        if i == len(y_true)-1 or y_score[i] != y_score[i+1]:
            recall = tp / total_pos
            precision = tp / (tp + fp)
            delta_recall = recall - prev_recall
            ap += precision * delta_recall
            prev_recall = recall

    return ap

def fast_numba_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Wrapper: sorts by score descending, casts y_true to 0/1 floats, then calls Numba core.
    """
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = (y_true[desc_idx] == 1).astype(np.float32)
    y_score_sorted = y_score[desc_idx].astype(np.float32)
    return fast_numba_auprc_nonw(y_true_sorted, y_score_sorted)





class ValMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, valid_dataset, test_label="val"):
        super().__init__()

        self.valid_dataset = valid_dataset
        self.test_label = test_label

    def on_epoch_end(self, epoch, logs=None):
        y_true_class_A, y_pred_class_A = [], []
        y_true_class_D, y_pred_class_D = [], []
        y_true_sse_A,   y_pred_sse_A   = [], []
        y_true_sse_all, y_pred_sse_all = [], []

        for x_batch, y_batch in self.valid_dataset:
            y_true_flat = y_batch.numpy().reshape(-1,4)
            mask = ~(y_true_flat[:,:3]==0).all(axis=1)
            cls = y_true_flat[mask,:3]
            reg = y_true_flat[mask,3]
            y_pred = self.model.predict(x_batch, verbose=0)
            if isinstance(y_pred, (list,tuple)):
                y_pred_cls, y_pred_sse = y_pred
                y_pred_sse = tf.nn.sigmoid(y_pred_sse).numpy().reshape(-1)[mask]
            else:
                y_pred_cls = y_pred.reshape(-1,3)[mask]
                y_pred_sse = None

            A_mask = cls[:,1]==1
            D_mask = cls[:,2]==1

            y_true_class_A.append(cls[A_mask,1])
            y_pred_class_A.append(y_pred_cls[A_mask,1])
            y_true_class_D.append(cls[D_mask,2])
            y_pred_class_D.append(y_pred_cls[D_mask,2])

            if y_pred_sse is not None:
                y_true_sse_A.append(reg[A_mask])
                y_pred_sse_A.append(y_pred_sse[A_mask])
                y_true_sse_all.append(reg)
                y_pred_sse_all.append(y_pred_sse)

        y_true_class_A = np.concatenate(y_true_class_A)
        y_pred_class_A = np.concatenate(y_pred_class_A)
        y_true_class_D = np.concatenate(y_true_class_D)
        y_pred_class_D = np.concatenate(y_pred_class_D)

        y_true_sse_A   = np.concatenate(y_true_sse_A)
        y_pred_sse_A   = np.concatenate(y_pred_sse_A)
        y_true_sse_all = np.concatenate(y_true_sse_all)
        y_pred_sse_all = np.concatenate(y_pred_sse_all)

        stats_A = print_topl_statistics(y_true_class_A, y_pred_class_A)
        stats_D = print_topl_statistics(y_true_class_D, y_pred_class_D)
        stats_reg_A = print_regression_statistics(y_true_sse_A, y_pred_sse_A)
        stats_reg_all = print_regression_statistics(y_true_sse_all, y_pred_sse_all)

        wandb_stats = {}
        for k,v in stats_A.items():
            wandb_stats[f"{self.test_label}_A_{k}"] = v
        for k,v in stats_D.items():
            wandb_stats[f"{self.test_label}_D_{k}"] = v
        for k,v in stats_reg_A.items():
            wandb_stats[f"{self.test_label}_regA_{k}"] = v
        for k,v in stats_reg_all.items():
            wandb_stats[f"{self.test_label}_regAll_{k}"] = v


        wandb.log(wandb_stats, step=epoch)

