import time
import os
from pathlib import Path
from collections import Counter
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import compute_class_weight

from scipy.stats import kruskal, mannwhitneyu, shapiro, levene, f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests

# -------------------- CONFIG (edit these) -----------------------------
BASE_DIR      = Path("")
NPZ_DIR       = BASE_DIR / "subject_windows_confounds_old_30sec"  # Path to directory with subject window NPZ files
K_STATES_LIST = list(range(4, 10))
N_SPLITS      = 10
SEEDS         = [100, 200, 300, 400, 500] # Seeds for the stable OBSERVED score
NUM_OF_STATES = 1
TR            = 0.8
WIN_SEC = 60
OVERLAP_FRAC = 0.75
WIN_LEN = int(round(WIN_SEC / TR))          # 75
WIN_STEP    = int(round(WIN_LEN * (1.0 - OVERLAP_FRAC)))  # 19

STEP_SEC      = WIN_STEP * TR

# Statistical test parameters
NORMALITY_ALPHA = 0.05  # Significance level for normality tests
VARIANCE_ALPHA = 0.05   # Significance level for variance homogeneity test
MIN_SAMPLE_SIZE = 3     # Minimum samples per group for normality testing

# Weighted averaging parameters
P_VALUE_SMOOTHING = 1e-10  # Small value to avoid division by zero
WEIGHT_POWER = 2.0         # Power to raise inverse p-values to (higher = more extreme weighting)

# --- OPTIMIZATION: More permutations for reliability, more jobs for speed ---
n_permutations = 1000   # <-- Increased for a reliable p-value
n_jobs = 10             # <-- Use all available CPU cores for max speed
random_state = 123
verbose = True
# ---------------------------------------------------------------------

# tasks to run
TASKS = ['0_vs_2', '0_vs_1', '1_vs_2', '0_vs_1and2']
task_readable = {
    '0_vs_2': 'N_vs_AP+',
    '0_vs_1': 'N_vs_AP-',
    '1_vs_2': 'AP-_vs_AP+',
    '0_vs_1and2': 'N_vs_APcombined'
}

# mapping function
def get_task_mapping(task):
    if task == '1_vs_2':
        return [1,2], {1:0,2:1}
    elif task == '0_vs_2':
        return [0,2], {0:0,2:1}
    elif task == '0_vs_1':
        return [0,1], {0:0,1:1}
    elif task == '0_vs_1and2':
        return [0,1,2], {0:0,1:1,2:1}
    else:
        raise ValueError(f"Unknown task: {task}")

# helper: dwell times
def dwell_times(preds, k):
    return np.bincount(preds, minlength=k) * STEP_SEC

# ─── STATISTICAL TESTING FUNCTION WITH NORMALITY CHECKS ────────────────────
def test_normality_and_variance(groups, state_id, fold_id):
    """
    Test normality and homogeneity of variance for groups.
    Returns: (use_parametric, test_details)
    """
    test_details = {
        'state': state_id,
        'fold': fold_id,
        'n_groups': len(groups),
        'group_sizes': [len(g) for g in groups],
        'normality_test': 'Not performed',
        'variance_test': 'Not performed',
        'use_parametric': False,
        'reason': ''
    }
    
    # Check if we have enough data for normality testing
    if any(len(g) < MIN_SAMPLE_SIZE for g in groups):
        test_details['reason'] = f'Sample size too small (min {MIN_SAMPLE_SIZE} required)'
        test_details['use_parametric'] = False
        return False, test_details
    
    # Test normality for each group using Shapiro-Wilk
    normality_pvals = []
    for i, group in enumerate(groups):
        if len(group) >= 3:  # Shapiro-Wilk requires at least 3 samples
            try:
                _, p_val = shapiro(group)
                normality_pvals.append(p_val)
            except Exception:
                normality_pvals.append(0.0)  # Assume non-normal if test fails
        else:
            normality_pvals.append(0.0)  # Too small for normality test
    
    # Check if all groups are normally distributed
    all_normal = all(p > NORMALITY_ALPHA for p in normality_pvals)
    test_details['normality_test'] = f"Shapiro-Wilk p-values: {[f'{p:.4f}' for p in normality_pvals]}"
    
    if not all_normal:
        test_details['reason'] = 'One or more groups not normally distributed'
        test_details['use_parametric'] = False
        return False, test_details
    
    # Test homogeneity of variance using Levene's test
    try:
        _, variance_p = levene(*groups)
        test_details['variance_test'] = f"Levene's test p-value: {variance_p:.4f}"
        
        if variance_p <= VARIANCE_ALPHA:
            test_details['reason'] = 'Unequal variances detected'
            test_details['use_parametric'] = False
            return False, test_details
        
    except Exception:
        test_details['variance_test'] = 'Levene test failed'
        test_details['reason'] = 'Could not test variance homogeneity'
        test_details['use_parametric'] = False
        return False, test_details
    
    # All assumptions met
    test_details['reason'] = 'All parametric assumptions satisfied'
    test_details['use_parametric'] = True
    return True, test_details

def choose_statistical_test_adaptive(groups, state_id, fold_id):
    """
    Choose appropriate statistical test based on normality and variance assumptions.
    Returns: (test_type, p_value, statistic, test_details)
    """
    # Basic checks
    if len(groups) < 2 or any(len(g) == 0 for g in groups):
        test_details = {
            'state': state_id, 'fold': fold_id, 'n_groups': len(groups),
            'group_sizes': [len(g) for g in groups], 'test_used': 'None',
            'reason': 'Insufficient groups or empty groups', 'use_parametric': False
        }
        return 'None', 1.0, 0.0, test_details
    
    # Convert to arrays if needed
    groups = [np.asarray(g) for g in groups]
    
    if np.std(np.concatenate(groups)) <= 1e-10:
        test_details = {
            'state': state_id, 'fold': fold_id, 'n_groups': len(groups),
            'group_sizes': [len(g) for g in groups], 'test_used': 'None',
            'reason': 'No variance in data', 'use_parametric': False
        }
        return 'None', 1.0, 0.0, test_details
    
    # Check if any group has zero variance (all identical values)
    group_stds = [np.std(g) for g in groups]
    if any(std <= 1e-10 for std in group_stds):
        test_details = {
            'state': state_id, 'fold': fold_id, 'n_groups': len(groups),
            'group_sizes': [len(g) for g in groups], 'test_used': 'None',
            'reason': 'One or more groups has no variance', 'use_parametric': False
        }
        return 'None', 1.0, 0.0, test_details
    
    # Test assumptions and choose appropriate test
    use_parametric, test_details = test_normality_and_variance(groups, state_id, fold_id)
    
    try:
        if use_parametric:
            # Use parametric tests
            if len(groups) == 2:
                # For 2 groups, use t-test
                statistic, p_value = ttest_ind(*groups, equal_var=True)
                test_details['test_used'] = 't-test'
            else:
                # Use ANOVA (F-test) for >2 groups
                statistic, p_value = f_oneway(*groups)
                test_details['test_used'] = 'ANOVA'
            
            return test_details['test_used'], p_value, statistic, test_details
        else:
            # Use non-parametric tests
            if len(groups) == 2:
                # Use Mann-Whitney U test for 2 groups
                statistic, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                test_details['test_used'] = 'Mann-Whitney'
                return 'Mann-Whitney', p_value, statistic, test_details
            else:
                # Use Kruskal-Wallis for >2 groups
                statistic, p_value = kruskal(*groups)
                test_details['test_used'] = 'Kruskal-Wallis'
                return 'Kruskal-Wallis', p_value, statistic, test_details
                
    except Exception as e:
        test_details['reason'] = f'Statistical test failed: {str(e)}'
        test_details['test_used'] = 'None'
        return 'None', 1.0, 0.0, test_details

# ─── WEIGHTED ENSEMBLE FUNCTIONS ───────────────────────────────────────────
def compute_pvalue_weights(p_values, smoothing=P_VALUE_SMOOTHING, power=WEIGHT_POWER):
    """
    Compute weights based on p-values. Lower p-values get higher weights.
    """
    # Add smoothing to avoid division by zero
    smoothed_p_values = np.array(p_values) + smoothing
    
    # Compute inverse p-values raised to power
    inverse_p = (1.0 / smoothed_p_values) ** power
    
    # Normalize to sum to 1
    weights = inverse_p / np.sum(inverse_p)
    
    return weights

def weighted_prediction(predictions, probabilities, weights):
    """
    Make weighted ensemble predictions.
    """
    # Weighted probability scores
    weighted_prob = np.average(probabilities, axis=1, weights=weights)
    
    # Hard predictions based on weighted probabilities
    weighted_pred = (weighted_prob > 0.5).astype(int)
    
    return weighted_pred, weighted_prob

# --- MODIFIED: Updated pipeline to use weighted ensemble method ---
def run_full_pipeline_for_subjects(subj_ids, windows, labels_vector, seeds_to_run):
    """
    subj_ids: array-like subject ids (length n_subjects)
    windows: list of arrays per subject (each array = stacked windows for that subject)
    labels_vector: binary labels (0/1) aligned with subj_ids/windows
    seeds_to_run: A list of random seeds to use for the ensemble.
    returns dict {'auc':..., 'mean_beta_per_sec':..., 'betas_vec': [...]}
    """
    rows_coef = []
    rows_auc_seed = []

    n_subjects = len(subj_ids)
    
    # Validate inputs
    if n_subjects == 0:
        return {"auc": float('nan'), "mean_beta_per_sec": float('nan'), "betas_vec": []}
    
    unique_labels = np.unique(labels_vector)
    if len(unique_labels) < 2:
        return {"auc": float('nan'), "mean_beta_per_sec": float('nan'), "betas_vec": []}
    
    for seed in seeds_to_run:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        y_true_all_ens, y_score_all_ens = [], []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(np.arange(n_subjects), labels_vector)):
            n_te = len(te_idx)
            y_pred_mat = np.empty((n_te, len(K_STATES_LIST)), dtype=int)
            y_prob_mat = np.zeros((n_te, len(K_STATES_LIST)), dtype=float)
            fold_p_values = []  # Store p-values for this fold to compute weights

            for j, k in enumerate(K_STATES_LIST):
                # fit KMeans on all train windows concatenated
                X_tr_win = np.vstack([windows[i] for i in tr_idx])
                km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
                km.fit(X_tr_win)

                dwell_tr = np.vstack([dwell_times(km.predict(windows[i]), k) for i in tr_idx])
                dwell_te  = np.vstack([dwell_times(km.predict(windows[i]), k) for i in te_idx])
                y_tr, y_te = labels_vector[tr_idx], labels_vector[te_idx]

                # --- FIXED: Get unique labels from training set only ---
                unique_train_labels = np.unique(y_tr)
                
                # --- UPDATED: Use adaptive statistical testing for state selection ---
                pvals = []
                test_types = []
                
                for s in range(k):
                    groups = [dwell_tr[y_tr == label, s] for label in unique_train_labels]
                    # Filter out empty groups
                    groups = [g for g in groups if len(g) > 0]
                    
                    if len(groups) < 2:
                        # Not enough groups for comparison
                        pvals.append(1.0)
                        test_types.append('None')
                    else:
                        test_type, p_value, statistic, test_details = choose_statistical_test_adaptive(
                            groups, s, fold)
                        pvals.append(p_value)
                        test_types.append(test_type)
                
                pvals = np.array(pvals, dtype=float)

                # Correct for multiple comparisons (within this fold & K)
                reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
                
                best_states = np.argsort(pvals_corrected)[:NUM_OF_STATES]  # length 1
                state_idx = int(np.atleast_1d(best_states)[0])
                
                # Store the best p-value for this K
                best_p_value = pvals_corrected[state_idx]
                fold_p_values.append(best_p_value)
                
                X_feat_tr = dwell_tr[:, best_states]
                X_feat_te = dwell_te[:, best_states]

                clf = make_pipeline(
                        MinMaxScaler(),
                        LogisticRegression(
                            penalty=None, solver="lbfgs",
                            class_weight="balanced",
                            C=1.0, tol=1e-3, max_iter=100, 
                            random_state=seed
                        )
                    )
                clf.fit(X_feat_tr, y_tr)

                logreg = clf.named_steps["logisticregression"]
                scaler = clf.named_steps["minmaxscaler"]
                beta_scaled = float(logreg.coef_[0,0])
                rng = float(scaler.data_range_[0])
                if rng == 0.0:
                    beta_per_sec = np.nan
                else:
                    beta_per_sec = beta_scaled / rng
                rows_coef.append(beta_per_sec)

                # preds
                y_prob = clf.predict_proba(X_feat_te)[:,1]
                y_pred = clf.predict(X_feat_te)
                y_pred_mat[:, j] = y_pred
                y_prob_mat[:, j] = y_prob

            # --- UPDATED: Use weighted ensemble instead of simple average ---
            # Compute weights based on p-values for this fold
            fold_weights = compute_pvalue_weights(fold_p_values)
            
            # Weighted ensemble across k for this fold
            y_pred_ens_weighted, y_score_ens_weighted = weighted_prediction(
                y_pred_mat, y_prob_mat, fold_weights
            )

            y_true_all_ens.extend(labels_vector[te_idx])
            y_score_all_ens.extend(y_score_ens_weighted)

        # per-seed AUC
        try:
            auc_seed = roc_auc_score(y_true_all_ens, y_score_all_ens)
        except Exception:
            auc_seed = float('nan')
        rows_auc_seed.append(auc_seed)

    ensemble_auc = float(np.nanmean(rows_auc_seed)) if rows_auc_seed else float('nan')
    betas = np.array(rows_coef, dtype=float)
    mean_beta_per_sec = float(np.nanmean(betas)) if np.isfinite(betas).any() else float('nan')
    return {"auc": ensemble_auc, "mean_beta_per_sec": mean_beta_per_sec, "betas_vec": rows_coef}

# function to run permutations for a single task
def run_task_permutations(task, n_permutations, n_jobs, base_dir, npz_dir, random_state, verbose=True):
    included_labels, label_map = get_task_mapping(task)
    readable = task_readable.get(task, task)
    out_nulls = base_dir / f"permutation_{readable}_weighted_adaptive_nulls.csv"
    out_summary = base_dir / f"permutation_{readable}_weighted_adaptive_summary.txt"

    # load NPZs and filter by included_labels
    subj_ids_list = []
    labels_raw_list = []
    windows_list = []

    for npz in sorted(npz_dir.glob("*_windows.npz")):
        data = np.load(npz, allow_pickle=True)
        lab_array = data.get("label", None)
        if lab_array is None: continue
        raw_lab = int(lab_array.item()) if isinstance(lab_array, np.ndarray) else int(lab_array)
        if raw_lab not in included_labels: continue
        
        subj_id = npz.stem.split("_")[0]
        subj_ids_list.append(subj_id)
        labels_raw_list.append(raw_lab)
        if "PA" in data and "AP" in data:
            windows_list.append(np.vstack([data["PA"], data["AP"]]))
        else:
            raise RuntimeError(f"{npz.name} missing PA/AP arrays")

    if not subj_ids_list:
        raise RuntimeError(f"No subjects found for task {task} in {npz_dir}")

    labels_bin = np.array([label_map[r] for r in labels_raw_list], dtype=int)
    subj_ids = np.array(subj_ids_list)
    windows = windows_list
    n_subjects = len(subj_ids)

    # --- ADDED: Validation for combined task ---
    if verbose:
        print(f"\n=== TASK {task} ({readable}) ===")
        print(f"Loaded {n_subjects} subjects")
        print(f"Binary label counts: {dict(Counter(labels_bin))}")
        if task == '0_vs_1and2':
            orig_counts = Counter(labels_raw_list)
            print(f"Original label distribution: {dict(orig_counts)}")
            # Validate that we have subjects from each expected group
            expected_labels = set(included_labels)
            found_labels = set(orig_counts.keys())
            missing = expected_labels - found_labels
            if missing:
                print(f"WARNING: Missing subjects with labels: {missing}")
        print("Using weighted ensemble with adaptive statistical testing")

    # --- ADDED: Check for minimum viable samples ---
    label_counts = Counter(labels_bin)
    min_class_count = min(label_counts.values())
    if min_class_count < N_SPLITS:
        print(f"WARNING: Smallest class has {min_class_count} samples, less than N_SPLITS={N_SPLITS}")
        print(f"  Reducing N_SPLITS to {min_class_count} for this task")
        effective_n_splits = min_class_count
    else:
        effective_n_splits = N_SPLITS

    # --- OPTIMIZATION: Observed run uses the full ensemble for a stable score ---
    t0 = time.time()
    observed = run_full_pipeline_for_subjects(subj_ids, windows, labels_bin, seeds_to_run=SEEDS)
    t_obs = time.time() - t0
    if verbose:
        print(f"Observed run time: {t_obs/60:.2f} min. Observed AUC = {observed['auc']:.4f}, mean_beta_per_sec = {observed['mean_beta_per_sec']:.6f}")

    # prepare permutations
    rng = np.random.RandomState(random_state)
    permutation_indices = [rng.permutation(n_subjects) for _ in range(n_permutations)]

    # --- OPTIMIZATION: Worker runs with only ONE seed for speed ---
    def _single_perm(perm_idx_tuple):
        i, perm_idx = perm_idx_tuple
        try:
            labels_perm = labels_bin[perm_idx]
            # Use the permutation number `i` as the single seed for this run
            res = run_full_pipeline_for_subjects(subj_ids, windows, labels_perm, seeds_to_run=[i])
            return (res["auc"], res["mean_beta_per_sec"], None)
        except Exception as e:
            return (np.nan, np.nan, repr(e))

    # run permutations (parallel when n_jobs>1)
    if verbose:
        # Estimate time for a single-seed run (roughly observed_time / num_seeds)
        t_perm_est = t_obs / len(SEEDS) if SEEDS else t_obs
        print(f"Starting {n_permutations} permutations with n_jobs={n_jobs} ... (approx. per-perm run ~ {t_perm_est:.1f} sec)")

    perm_results = []
    try:
        if n_jobs == 1:
            for i, perm_idx in enumerate(permutation_indices):
                if verbose and ((i+1) % max(1, n_permutations // 10) == 0):
                    print(f"  Perm {i+1}/{n_permutations}")
                perm_results.append(_single_perm((i, perm_idx)))
        else:
            # use joblib with verbosity to print progress
            parallel_verbosity = 5 if verbose else 0
            perm_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=parallel_verbosity)(
                delayed(_single_perm)((i, perm_idx)) for i, perm_idx in enumerate(permutation_indices)
            )
    except Exception as e:
        # fallback to serial if parallel fails
        print(f"Parallel execution failed; falling back to serial. Exception: {e}")
        perm_results = [_single_perm((i, p_idx)) for i, p_idx in enumerate(permutation_indices)]

    # unpack results and record exceptions
    perm_aucs = np.array([r[0] for r in perm_results], dtype=float)
    perm_betas = np.array([r[1] for r in perm_results], dtype=float)
    perm_errs  = [r[2] for r in perm_results]

    # empirical p-values (add-one correction)
    obs_auc = observed["auc"]
    obs_beta = observed["mean_beta_per_sec"]
    
    # --- FIXED: Handle NaN values in permutation results ---
    valid_aucs = perm_aucs[~np.isnan(perm_aucs)]
    valid_betas = perm_betas[~np.isnan(perm_betas)]
    
    if len(valid_aucs) > 0:
        p_auc_emp = (np.sum(valid_aucs >= obs_auc) + 1) / (len(valid_aucs) + 1)
    else:
        p_auc_emp = np.nan
        
    if np.isnan(obs_beta) or len(valid_betas) == 0:
        p_beta_one = np.nan
        p_beta_two = np.nan
    else:
        if obs_beta < 0:
            p_beta_one = (np.sum(valid_betas <= obs_beta) + 1) / (len(valid_betas) + 1)
        else:
            p_beta_one = (np.sum(valid_betas >= obs_beta) + 1) / (len(valid_betas) + 1)
        p_beta_two = (np.sum(np.abs(valid_betas) >= abs(obs_beta)) + 1) / (len(valid_betas) + 1)

    # save nulls and summary
    df_null = pd.DataFrame({"perm_auc": perm_aucs, "perm_beta_per_sec": perm_betas, "error": perm_errs})
    df_null.to_csv(out_nulls, index=False)

    with open(out_summary, "w") as fh:
        fh.write(f"TASK: {task} ({readable})\n")
        fh.write(f"METHOD: Weighted ensemble with adaptive statistical testing\n")
        fh.write(f"n_subjects: {n_subjects}\n")
        fh.write(f"binary_counts: {dict(Counter(labels_bin))}\n")
        if task == '0_vs_1and2':
            fh.write(f"original_label_counts: {dict(Counter(labels_raw_list))}\n")
        fh.write(f"observed_auc: {obs_auc:.6f}\n")
        fh.write(f"observed_mean_beta_per_sec: {obs_beta:.6f}\n")
        fh.write(f"observed_run_time_sec: {t_obs:.2f}\n")
        fh.write(f"n_permutations: {n_permutations}\n")
        fh.write(f"n_valid_permutations: {len(valid_aucs)}\n")
        fh.write(f"n_jobs: {n_jobs}\n")
        fh.write(f"weighting_power: {WEIGHT_POWER}\n")
        fh.write(f"p_value_smoothing: {P_VALUE_SMOOTHING}\n")
        fh.write("\nEMPIRICAL P-VALUES (add-one correction):\n")
        fh.write(f"p_auc_empirical_one_sided (perm >= obs): {p_auc_emp:.6e}\n")
        fh.write(f"p_beta_empirical_one_sided (direction-aware): {p_beta_one:.6e}\n")
        fh.write(f"p_beta_empirical_two_sided: {p_beta_two:.6e}\n")
        fh.write("\nPermuted AUC summary: mean={:.4f}, std={:.4f}, max={:.4f}\n".format(
            np.nanmean(perm_aucs), np.nanstd(perm_aucs), np.nanmax(perm_aucs)))
        fh.write("Permuted beta_per_sec summary: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}\n".format(
            np.nanmean(perm_betas), np.nanstd(perm_betas), np.nanmin(perm_betas), np.nanmax(perm_betas)))
        n_err = sum(1 for e in perm_errs if e is not None)
        fh.write(f"\npermutation_errors: {n_err}\n")
        if n_err > 0:
            fh.write("Some permutation runs raised exceptions; see nulls file 'error' column for details.\n")

    if verbose:
        print(f"Saved null distributions to {out_nulls}")
        print(f"Saved summary to {out_summary}")

    return {"task": task, "readable": readable, "observed": observed, "perm_aucs": perm_aucs, "perm_betas": perm_betas,
            "p_auc_emp": p_auc_emp, "p_beta_one": p_beta_one, "p_beta_two": p_beta_two,
            "out_nulls": out_nulls, "out_summary": out_summary}

# -------------------- Run for all tasks ---------------------------------
if __name__ == "__main__":
    overall_start = time.time()
    results_all = []
    
    print("Running permutation tests with:")
    print(f"  - Weighted ensemble averaging (power={WEIGHT_POWER})")
    print(f"  - Adaptive statistical testing (ANOVA/t-test vs Mann-Whitney/Kruskal-Wallis)")
    print(f"  - Normality testing (Shapiro-Wilk, α={NORMALITY_ALPHA})")
    print(f"  - Variance homogeneity testing (Levene's test, α={VARIANCE_ALPHA})")
    
    for task in TASKS:
        try:
            res = run_task_permutations(task, n_permutations=n_permutations, n_jobs=n_jobs,
                                        base_dir=BASE_DIR, npz_dir=NPZ_DIR, random_state=random_state, verbose=verbose)
            results_all.append(res)
        except Exception as e:
            print(f"Task {task} failed with exception:\n{traceback.format_exc()}")

    total_time = time.time() - overall_start
    print(f"\nAll tasks finished. Total wall time: {total_time/60:.2f} minutes.")
    
    # summarize quick table
    rows = []
    for r in results_all:
        rows.append({
            "task": r["task"],
            "readable": r["readable"],
            "observed_auc": r["observed"]["auc"],
            "observed_beta_per_sec": r["observed"]["mean_beta_per_sec"],
            "p_auc_emp": r["p_auc_emp"],
            "p_beta_one": r["p_beta_one"],
            "out_nulls": str(r["out_nulls"])
        })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(BASE_DIR / "permutation_all_tasks_weighted_adaptive_summary.csv", index=False)
    print("Wrote overall summary to:", BASE_DIR / "permutation_all_tasks_weighted_adaptive_summary.csv")