import time
import os
from pathlib import Path
from collections import defaultdict
import traceback

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)

from scipy.stats import kruskal, mannwhitneyu, shapiro, levene, f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests

# -------------------- CONFIG (same as permutation_f1.py) ----------------------
BASE_DIR      = Path("")
NPZ_DIR       = BASE_DIR / "subject_windows_confounds_old_60sec"
K_STATES_LIST = list(range(4, 10))
N_SPLITS      = 10
SEEDS         = [100, 200, 300, 400, 500]
NUM_OF_STATES = 1
TR            = 0.8
WIN_SEC = 60
OVERLAP_FRAC = 0.75
WIN_LEN = int(round(WIN_SEC / TR))
WIN_STEP = int(round(WIN_LEN * (1.0 - OVERLAP_FRAC)))
STEP_SEC = WIN_STEP * TR  # 6.4 seconds per step with 90% overlap

# Time scales for interpretation
WINDOW_SEC    = 60.0              # your window length
MINUTE_SEC    = 60.0

# Statistical test parameters
NORMALITY_ALPHA = 0.05
VARIANCE_ALPHA = 0.05
MIN_SAMPLE_SIZE = 3

# Weighted averaging parameters
P_VALUE_SMOOTHING = 1e-10
WEIGHT_POWER = 2.0

verbose = True

# tasks to run
TASKS = ['0_vs_1', '0_vs_2', '1_vs_2', '0_vs_1and2']
task_readable = {
    '0_vs_1and2': 'N_vs_APcombined',
    '0_vs_2': 'N_vs_AP+',
    '0_vs_1': 'N_vs_AP-',
    '1_vs_2': 'AP-_vs_AP+',
}

task_readable_latex = {
    '0_vs_1': 'N vs. A+P-',
    '0_vs_2': 'N vs. A+P+',
    '1_vs_2': 'A+P- vs. A+P+',
    '0_vs_1and2': 'N vs (A+P- & A+P+)',
}

# ------------------------------------------------------------------------------

def get_task_mapping(task):
    if task == '1_vs_2':
        return [1,2], {1:0, 2:1}
    elif task == '0_vs_2':
        return [0,2], {0:0, 2:1}
    elif task == '0_vs_1':
        return [0,1], {0:0, 1:1}
    elif task == '0_vs_1and2':
        return [0,1,2], {0:0, 1:1, 2:1}
    else:
        raise ValueError(f"Unknown task: {task}")

def dwell_times(preds, k):
    return np.bincount(preds, minlength=k) * STEP_SEC

def test_normality_and_variance(groups, state_id, fold_id):
    test_details = {
        'state': state_id, 'fold': fold_id, 'n_groups': len(groups),
        'group_sizes': [len(g) for g in groups],
        'normality_test': 'Not performed', 'variance_test': 'Not performed',
        'use_parametric': False, 'reason': ''
    }
    
    if any(len(g) < MIN_SAMPLE_SIZE for g in groups):
        test_details['reason'] = f'Sample size too small (min {MIN_SAMPLE_SIZE} required)'
        return False, test_details
    
    normality_pvals = []
    for group in groups:
        if len(group) >= 3:
            try:
                _, p_val = shapiro(group)
                normality_pvals.append(p_val)
            except:
                normality_pvals.append(0.0)
        else:
            normality_pvals.append(0.0)
    
    all_normal = all(p > NORMALITY_ALPHA for p in normality_pvals)
    test_details['normality_test'] = f"Shapiro-Wilk p-values: {[f'{p:.4f}' for p in normality_pvals]}"
    
    if not all_normal:
        test_details['reason'] = 'One or more groups not normally distributed'
        return False, test_details
    
    try:
        _, variance_p = levene(*groups)
        test_details['variance_test'] = f"Levene's test p-value: {variance_p:.4f}"
        if variance_p <= VARIANCE_ALPHA:
            test_details['reason'] = 'Unequal variances detected'
            return False, test_details
    except:
        test_details['variance_test'] = 'Levene test failed'
        test_details['reason'] = 'Could not test variance homogeneity'
        return False, test_details
    
    test_details['reason'] = 'All parametric assumptions satisfied'
    test_details['use_parametric'] = True
    return True, test_details

def choose_statistical_test_adaptive(groups, state_id, fold_id):
    if len(groups) < 2 or any(g.size == 0 for g in groups):
        return 'None', 1.0, 0.0, {}
    
    if np.std(np.concatenate(groups)) <= 1e-10:
        return 'None', 1.0, 0.0, {}
    
    group_stds = [np.std(g) for g in groups]
    if any(std <= 1e-10 for std in group_stds):
        return 'None', 1.0, 0.0, {}
    
    use_parametric, test_details = test_normality_and_variance(groups, state_id, fold_id)
    
    try:
        if use_parametric:
            if len(groups) == 2:
                statistic, p_value = ttest_ind(*groups, equal_var=True)
                test_details['test_used'] = 't-test'
            else:
                statistic, p_value = f_oneway(*groups)
                test_details['test_used'] = 'ANOVA'
            return test_details['test_used'], p_value, statistic, test_details
        else:
            if len(groups) == 2:
                statistic, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                return 'Mann-Whitney', p_value, statistic, test_details
            else:
                statistic, p_value = kruskal(*groups)
                return 'Kruskal-Wallis', p_value, statistic, test_details
    except Exception:
        return 'None', 1.0, 0.0, test_details

def compute_pvalue_weights(p_values, smoothing=P_VALUE_SMOOTHING, power=WEIGHT_POWER):
    smoothed_p_values = np.array(p_values) + smoothing
    inverse_p = (1.0 / smoothed_p_values) ** power
    weights = inverse_p / np.sum(inverse_p)
    return weights

def weighted_prediction(predictions, probabilities, weights):
    weighted_prob = np.average(probabilities, axis=1, weights=weights)
    weighted_pred = (weighted_prob > 0.5).astype(int)
    return weighted_pred, weighted_prob

def run_full_pipeline_with_metrics(subj_ids, windows, labels_vector, seeds_to_run):
    """
    Run the full pipeline and collect:
    1. Performance metrics per seed (accuracy, precision, recall, f1, auc)
    2. Coefficient information (coef, SE, direction based on group means)
    """
    n_subjects = len(subj_ids)
    
    # Store metrics per seed
    seed_metrics = []
    
    # Store all coefficient info for direction-based analysis
    all_coef_info = []
    
    for seed in seeds_to_run:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        y_true_all_ens, y_pred_all_ens, y_prob_all_ens = [], [], []
        
        for fold, (tr_idx, te_idx) in enumerate(skf.split(np.arange(n_subjects), labels_vector)):
            n_te = len(te_idx)
            y_pred_mat = np.empty((n_te, len(K_STATES_LIST)), dtype=int)
            y_prob_mat = np.zeros((n_te, len(K_STATES_LIST)), dtype=float)
            fold_p_values = []
            
            for j, k in enumerate(K_STATES_LIST):
                # Fit KMeans
                X_tr_win = np.vstack([windows[i] for i in tr_idx])
                km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
                km.fit(X_tr_win)
                
                dwell_tr = np.vstack([dwell_times(km.predict(windows[i]), k) for i in tr_idx])
                dwell_te = np.vstack([dwell_times(km.predict(windows[i]), k) for i in te_idx])
                y_tr, y_te = labels_vector[tr_idx], labels_vector[te_idx]
                
                # Statistical testing for state selection
                pvals = []
                for s in range(k):
                    groups = [dwell_tr[y_tr == label, s] for label in np.unique(y_tr)]
                    _, p_value, _, _ = choose_statistical_test_adaptive(groups, s, fold)
                    pvals.append(p_value)
                
                pvals = np.array(pvals, dtype=float)
                reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
                
                best_states = np.argsort(pvals_corrected)[:NUM_OF_STATES]
                state_idx = int(np.atleast_1d(best_states)[0])
                best_p_value = pvals_corrected[state_idx]
                fold_p_values.append(best_p_value)
                
                X_feat_tr = dwell_tr[:, best_states]
                X_feat_te = dwell_te[:, best_states]
                
                # Compute group means for the selected state to determine direction
                mean_class0 = np.mean(dwell_tr[y_tr == 0, state_idx])
                mean_class1 = np.mean(dwell_tr[y_tr == 1, state_idx])
                # direction: 1 if class0 > class1 (class0 has higher dwell), else -1
                direction = 1 if mean_class0 > mean_class1 else -1
                
                # Fit classifier
                clf = make_pipeline(
                    MinMaxScaler(),
                    LogisticRegression(
                        penalty=None, solver="lbfgs",
                        class_weight="balanced",
                        C=1.0, tol=1e-3, max_iter=10000,
                        random_state=seed
                    )
                )
                clf.fit(X_feat_tr, y_tr)
                
                logreg = clf.named_steps["logisticregression"]
                scaler = clf.named_steps["minmaxscaler"]
                
                beta_scaled = float(logreg.coef_[0, 0])
                b0_scaled = float(logreg.intercept_[0])
                
                mn = float(scaler.data_min_[0])
                rng = float(scaler.data_range_[0])
                
                if rng == 0.0:
                    beta_per_sec = np.nan
                    coef_per_60s = np.nan
                    or_per_60s = np.nan
                else:
                    # Back-transform: logit(x) = b0_scaled + beta_scaled * (x - mn)/rng
                    #                          = (b0_scaled - beta_scaled*mn/rng) + (beta_scaled/rng) * x
                    beta_per_sec = beta_scaled / rng
                    
                    # Coefficient for 60 seconds of additional dwell time
                    coef_per_60s = beta_per_sec * MINUTE_SEC
                    or_per_60s = float(np.exp(coef_per_60s))
                
                # Store coefficient info
                all_coef_info.append({
                    'seed': seed,
                    'fold': fold,
                    'k': k,
                    'state_idx': state_idx,
                    'coef_per_sec': beta_per_sec,
                    'coef_per_60s': coef_per_60s,
                    'or_per_60s': or_per_60s,
                    'direction': direction,  # 1 = class0 higher, -1 = class1 higher
                    'mean_class0': mean_class0,
                    'mean_class1': mean_class1,
                })
                
                # Predictions
                y_prob = clf.predict_proba(X_feat_te)[:, 1]
                y_pred = clf.predict(X_feat_te)
                y_pred_mat[:, j] = y_pred
                y_prob_mat[:, j] = y_prob
            
            # Weighted ensemble
            fold_weights = compute_pvalue_weights(fold_p_values)
            y_pred_ens_weighted, y_prob_ens_weighted = weighted_prediction(
                y_pred_mat, y_prob_mat, fold_weights
            )
            
            y_true_all_ens.extend(labels_vector[te_idx])
            y_pred_all_ens.extend(y_pred_ens_weighted)
            y_prob_all_ens.extend(y_prob_ens_weighted)
        
        # Compute metrics for this seed
        y_true_arr = np.array(y_true_all_ens)
        y_pred_arr = np.array(y_pred_all_ens)
        y_prob_arr = np.array(y_prob_all_ens)
        
        try:
            acc = accuracy_score(y_true_arr, y_pred_arr)
            prec = precision_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
            rec = recall_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
            f1 = f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
            auc = roc_auc_score(y_true_arr, y_prob_arr)
        except Exception:
            acc = prec = rec = f1 = auc = np.nan
        
        seed_metrics.append({
            'seed': seed,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_macro': f1,
            'auc_roc': auc
        })
    
    return seed_metrics, all_coef_info


def analyze_coefficients_by_direction(coef_info_list, task):
    """
    Separate coefficients by direction and compute statistics.
    
    IMPORTANT: We first average within each seed to get independent observations,
    then compute statistics across seeds (n=5).
    
    Direction 1: class0 (N or AP-) has higher dwell time -> negative coefficient expected
    Direction -1: class1 (AP-, AP+, or combined) has higher dwell time -> positive coefficient expected
    
    Reports:
    - coef_per_60s: coefficient scaled to 60 seconds
    - or_per_60s: odds ratio for 60 seconds additional dwell time
    - STD across seeds
    - p-value from one-sample t-test against 0 (across seeds)
    """
    df = pd.DataFrame(coef_info_list)
    df = df.dropna(subset=['coef_per_60s'])
    
    results = {}
    
    # Direction 1: class0 has higher dwell (N higher for N vs AP tasks, AP- higher for AP- vs AP+)
    df_dir1 = df[df['direction'] == 1]
    
    if len(df_dir1) > 0:
        # First, average within each seed to get independent observations
        seed_means = df_dir1.groupby('seed').agg({
            'coef_per_60s': 'mean',
            'or_per_60s': 'mean'
        }).reset_index()
        
        n_seeds = len(seed_means)
        coef_values = seed_means['coef_per_60s'].values
        or_values = seed_means['or_per_60s'].values
        
        coef_mean = np.mean(coef_values)
        coef_std = np.std(coef_values, ddof=1) if n_seeds > 1 else 0.0
        or_mean = np.mean(or_values)
        or_std = np.std(or_values, ddof=1) if n_seeds > 1 else 0.0
        
        # One-sample t-test against 0 (now with proper n=5 seeds)
        if n_seeds > 1:
            t_stat, p_val = stats.ttest_1samp(coef_values, 0)
        else:
            p_val = 1.0
        
        results['direction1'] = {
            'count_total': len(df_dir1),  # Total fold×seed×k combinations
            'count_seeds': n_seeds,        # Number of seeds (independent obs)
            'coef_mean': coef_mean,
            'coef_std': coef_std,
            'or_mean': or_mean,
            'or_std': or_std,
            'p_value': p_val,
            'seed_coefficients': coef_values.tolist(),
            'seed_or': or_values.tolist()
        }
    else:
        results['direction1'] = None
    
    # Direction -1: class1 has higher dwell (AP groups higher for N vs AP, AP+ higher for AP- vs AP+)
    df_dir2 = df[df['direction'] == -1]
    
    if len(df_dir2) > 0:
        # First, average within each seed to get independent observations
        seed_means = df_dir2.groupby('seed').agg({
            'coef_per_60s': 'mean',
            'or_per_60s': 'mean'
        }).reset_index()
        
        n_seeds = len(seed_means)
        coef_values = seed_means['coef_per_60s'].values
        or_values = seed_means['or_per_60s'].values
        
        coef_mean = np.mean(coef_values)
        coef_std = np.std(coef_values, ddof=1) if n_seeds > 1 else 0.0
        or_mean = np.mean(or_values)
        or_std = np.std(or_values, ddof=1) if n_seeds > 1 else 0.0
        
        if n_seeds > 1:
            t_stat, p_val = stats.ttest_1samp(coef_values, 0)
        else:
            p_val = 1.0
        
        results['direction2'] = {
            'count_total': len(df_dir2),
            'count_seeds': n_seeds,
            'coef_mean': coef_mean,
            'coef_std': coef_std,
            'or_mean': or_mean,
            'or_std': or_std,
            'p_value': p_val,
            'seed_coefficients': coef_values.tolist(),
            'seed_or': or_values.tolist()
        }
    else:
        results['direction2'] = None
    
    return results


def run_task_analysis(task, base_dir, npz_dir, verbose=True):
    """Run full analysis for a single task."""
    included_labels, label_map = get_task_mapping(task)
    readable = task_readable.get(task, task)
    
    # Load data
    subj_ids_list, labels_raw_list, windows_list = [], [], []
    
    for npz in sorted(npz_dir.glob("*_windows.npz")):
        data = np.load(npz, allow_pickle=True)
        lab_array = data.get("label", None)
        if lab_array is None:
            continue
        raw_lab = int(lab_array.item()) if isinstance(lab_array, np.ndarray) else int(lab_array)
        if raw_lab not in included_labels:
            continue
        
        subj_id = npz.stem.split("_")[0]
        subj_ids_list.append(subj_id)
        labels_raw_list.append(raw_lab)
        if "PA" in data and "AP" in data:
            windows_list.append(np.vstack([data["PA"], data["AP"]]))
        else:
            raise RuntimeError(f"{npz.name} missing PA/AP arrays")
    
    if not subj_ids_list:
        raise RuntimeError(f"No subjects found for task {task}")
    
    labels_bin = np.array([label_map[r] for r in labels_raw_list], dtype=int)
    subj_ids = np.array(subj_ids_list)
    
    if verbose:
        print(f"\n=== TASK {task} ({readable}) ===")
        print(f"Loaded {len(subj_ids)} subjects; counts: {dict(zip(*np.unique(labels_bin, return_counts=True)))}")
    
    # Run pipeline
    t0 = time.time()
    seed_metrics, coef_info = run_full_pipeline_with_metrics(
        subj_ids, windows_list, labels_bin, seeds_to_run=SEEDS
    )
    t_elapsed = time.time() - t0
    
    if verbose:
        print(f"Completed in {t_elapsed/60:.2f} min")
    
    # Analyze coefficients by direction
    coef_analysis = analyze_coefficients_by_direction(coef_info, task)
    
    return {
        'task': task,
        'readable': readable,
        'seed_metrics': seed_metrics,
        'coef_info': coef_info,
        'coef_analysis': coef_analysis
    }


if __name__ == "__main__":
    overall_start = time.time()
    all_results = {}
    
    print("=" * 70)
    print("Computing observed performance metrics and coefficient statistics")
    print(f"Step size: {STEP_SEC:.1f}s, Window: {WIN_SEC}s, Overlap: {OVERLAP_FRAC*100:.0f}%")
    print("=" * 70)
    
    for task in TASKS:
        try:
            result = run_task_analysis(task, BASE_DIR, NPZ_DIR, verbose=verbose)
            all_results[task] = result
        except Exception as e:
            print(f"Task {task} failed:\n{traceback.format_exc()}")
    
    # =========================================================================
    # TABLE 1: Performance metrics (mean ± std)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TABLE 1: Performance Metrics")
    print("=" * 70)
    
    perf_rows = []
    for task in TASKS:
        if task not in all_results:
            continue
        result = all_results[task]
        df_metrics = pd.DataFrame(result['seed_metrics'])
        
        row = {
            'task': task,
            'readable': task_readable_latex.get(task, task),
            'accuracy_mean': df_metrics['accuracy'].mean(),
            'accuracy_std': df_metrics['accuracy'].std(),
            'precision_mean': df_metrics['precision'].mean(),
            'precision_std': df_metrics['precision'].std(),
            'recall_mean': df_metrics['recall'].mean(),
            'recall_std': df_metrics['recall'].std(),
            'f1_macro_mean': df_metrics['f1_macro'].mean(),
            'f1_macro_std': df_metrics['f1_macro'].std(),
            'auc_roc_mean': df_metrics['auc_roc'].mean(),
            'auc_roc_std': df_metrics['auc_roc'].std(),
        }
        perf_rows.append(row)
        
        print(f"\n{row['readable']}:")
        print(f"  Accuracy:  {row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}")
        print(f"  Precision: {row['precision_mean']:.3f} ± {row['precision_std']:.3f}")
        print(f"  Recall:    {row['recall_mean']:.3f} ± {row['recall_std']:.3f}")
        print(f"  F1 (macro):{row['f1_macro_mean']:.3f} ± {row['f1_macro_std']:.3f}")
        print(f"  AUC-ROC:   {row['auc_roc_mean']:.3f} ± {row['auc_roc_std']:.3f}")
    
    df_performance = pd.DataFrame(perf_rows)
    df_performance.to_csv(BASE_DIR / "observed_performance_metrics.csv", index=False)
    print(f"\nSaved to: {BASE_DIR / 'observed_performance_metrics.csv'}")
    
    # =========================================================================
    # TABLE 2 & 3: Coefficient statistics by direction (per 60 seconds)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TABLE 2: Coefficients (Direction 1 - N/AP- higher dwell) - per 60s")
    print("(Statistics computed across seeds, n=5)")
    print("=" * 70)
    
    dir1_rows = []
    dir2_rows = []
    
    for task in TASKS:
        if task not in all_results:
            continue
        result = all_results[task]
        coef_analysis = result['coef_analysis']
        readable = task_readable_latex.get(task, task)
        
        # Direction 1
        if coef_analysis['direction1'] is not None:
            d1 = coef_analysis['direction1']
            dir1_rows.append({
                'task': task,
                'readable': readable,
                'count_total': d1['count_total'],
                'count_seeds': d1['count_seeds'],
                'coef_mean': d1['coef_mean'],
                'coef_std': d1['coef_std'],
                'or_mean': d1['or_mean'],
                'or_std': d1['or_std'],
                'p_value': d1['p_value'],
                'seed_coefs': d1['seed_coefficients']
            })
            print(f"\n{readable}:")
            print(f"  Total obs: {d1['count_total']} (across {d1['count_seeds']} seeds)")
            print(f"  Coef (per 60s): {d1['coef_mean']:.3f} ± {d1['coef_std']:.3f}")
            print(f"  OR (per 60s):   {d1['or_mean']:.3f} ± {d1['or_std']:.3f}")
            print(f"  p (t-test, df={d1['count_seeds']-1}): {d1['p_value']:.4f}")
            print(f"  Per-seed coefs: {[f'{c:.3f}' for c in d1['seed_coefficients']]}")
        
        # Direction 2
        if coef_analysis['direction2'] is not None:
            d2 = coef_analysis['direction2']
            dir2_rows.append({
                'task': task,
                'readable': readable,
                'count_total': d2['count_total'],
                'count_seeds': d2['count_seeds'],
                'coef_mean': d2['coef_mean'],
                'coef_std': d2['coef_std'],
                'or_mean': d2['or_mean'],
                'or_std': d2['or_std'],
                'p_value': d2['p_value'],
                'seed_coefs': d2['seed_coefficients']
            })
    
    # Apply FDR correction across all direction 1 p-values
    if dir1_rows:
        pvals_dir1 = [r['p_value'] for r in dir1_rows]
        _, pvals_fdr1, _, _ = multipletests(pvals_dir1, alpha=0.05, method='fdr_bh')
        for i, row in enumerate(dir1_rows):
            row['p_fdr'] = pvals_fdr1[i]
    
    if dir2_rows:
        pvals_dir2 = [r['p_value'] for r in dir2_rows]
        _, pvals_fdr2, _, _ = multipletests(pvals_dir2, alpha=0.05, method='fdr_bh')
        for i, row in enumerate(dir2_rows):
            row['p_fdr'] = pvals_fdr2[i]
    
    df_dir1 = pd.DataFrame(dir1_rows)
    df_dir2 = pd.DataFrame(dir2_rows)
    
    df_dir1.to_csv(BASE_DIR / "coef_stats_direction1_N_higher.csv", index=False)
    df_dir2.to_csv(BASE_DIR / "coef_stats_direction2_AP_higher.csv", index=False)
    
    print("\n" + "=" * 70)
    print("TABLE 3: Coefficients (Direction 2 - AP/AP+ higher dwell) - per 60s")
    print("(Statistics computed across seeds, n=5)")
    print("=" * 70)
    for row in dir2_rows:
        print(f"\n{row['readable']}:")
        print(f"  Total obs: {row['count_total']} (across {row['count_seeds']} seeds)")
        print(f"  Coef (per 60s): {row['coef_mean']:.3f} ± {row['coef_std']:.3f}")
        print(f"  OR (per 60s):   {row['or_mean']:.3f} ± {row['or_std']:.3f}")
        print(f"  p (t-test, df={row['count_seeds']-1}): {row['p_value']:.4f}, p(FDR): {row['p_fdr']:.4f}")
        print(f"  Per-seed coefs: {[f'{c:.3f}' for c in row['seed_coefs']]}")
    
    print(f"\nSaved to:")
    print(f"  {BASE_DIR / 'coef_stats_direction1_N_higher.csv'}")
    print(f"  {BASE_DIR / 'coef_stats_direction2_AP_higher.csv'}")
    
    # =========================================================================
    # Save all detailed coefficient info for further analysis
    # =========================================================================
    all_coef_rows = []
    for task in TASKS:
        if task not in all_results:
            continue
        for info in all_results[task]['coef_info']:
            info['task'] = task
            info['readable'] = task_readable_latex.get(task, task)
            all_coef_rows.append(info)
    
    df_all_coef = pd.DataFrame(all_coef_rows)
    df_all_coef.to_csv(BASE_DIR / "all_coefficient_details.csv", index=False)
    print(f"\nSaved detailed coefficients to: {BASE_DIR / 'all_coefficient_details.csv'}")
    
    # =========================================================================
    # Save per-seed metrics for detailed analysis
    # =========================================================================
    all_seed_rows = []
    for task in TASKS:
        if task not in all_results:
            continue
        for metrics in all_results[task]['seed_metrics']:
            metrics['task'] = task
            metrics['readable'] = task_readable_latex.get(task, task)
            all_seed_rows.append(metrics)
    
    df_all_seeds = pd.DataFrame(all_seed_rows)
    df_all_seeds.to_csv(BASE_DIR / "all_seed_metrics.csv", index=False)
    print(f"Saved per-seed metrics to: {BASE_DIR / 'all_seed_metrics.csv'}")
    
    total_time = time.time() - overall_start
    print(f"\nTotal time: {total_time/60:.2f} minutes")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)