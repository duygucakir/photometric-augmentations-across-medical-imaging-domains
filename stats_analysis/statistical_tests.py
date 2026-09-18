"""
Phase 3 - Statistical Analysis Module
======================================
Paired t-tests, Holm–Bonferroni correction, Cohen's d,
95% confidence intervals for ΔF1, AUC, and balanced accuracy.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from itertools import combinations

logger = logging.getLogger(__name__)


# =============================================================================
# Effect Sizes
# =============================================================================

def cohens_d_paired(x1, x2):
    """
    Compute Cohen's d for paired samples.
    
    d = mean(x1 - x2) / std(x1 - x2)
    """
    diff = np.array(x1) - np.array(x2)
    if np.std(diff) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def confidence_interval(values, confidence=0.95):
    """Compute confidence interval for a set of values."""
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean - h), float(mean + h)


def delta_confidence_interval(x1, x2, confidence=0.95):
    """Compute 95% CI for the difference (x1 - x2)."""
    diff = np.array(x1) - np.array(x2)
    return confidence_interval(diff, confidence)


# =============================================================================
# Statistical Tests
# =============================================================================

def paired_ttest(x1, x2):
    """
    Paired t-test between two conditions across folds.
    
    Returns:
        t_stat, p_value
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    if len(x1) != len(x2):
        raise ValueError(f"Unequal lengths: {len(x1)} vs {len(x2)}")
    
    if np.allclose(x1, x2):
        return 0.0, 1.0
    
    t_stat, p_value = stats.ttest_rel(x1, x2)
    return float(t_stat), float(p_value)


def holm_bonferroni_correction(p_values):
    """
    Apply Holm–Bonferroni correction to a list of p-values.
    
    Args:
        p_values: list of (comparison_name, p_value) tuples
    
    Returns:
        list of (comparison_name, original_p, corrected_p, significant) tuples
    """
    n = len(p_values)
    # Sort by p-value
    sorted_pvals = sorted(p_values, key=lambda x: x[1])
    
    results = []
    for rank, (name, p) in enumerate(sorted_pvals, 1):
        corrected_p = min(p * (n - rank + 1), 1.0)
        significant = corrected_p < 0.05
        results.append({
            'comparison': name,
            'p_value': float(p),
            'corrected_p': float(corrected_p),
            'rank': rank,
            'significant': bool(significant),
        })
    
    return results


# =============================================================================
# Full Statistical Analysis
# =============================================================================

def run_pairwise_analysis(fold_results, baseline_key='baseline_none', 
                          metrics_to_test=None):
    """
    Run pairwise statistical analysis comparing all conditions to baseline.
    
    Args:
        fold_results: dict of condition_name -> {metric: [fold1, fold2, ...]}
        baseline_key: name of baseline condition
        metrics_to_test: list of metrics to test
    
    Returns:
        analysis_results: comprehensive statistical analysis
    """
    if metrics_to_test is None:
        metrics_to_test = ['macro_f1', 'auc', 'balanced_accuracy']
    
    if baseline_key not in fold_results:
        logger.warning(f"Baseline '{baseline_key}' not found in results")
        return {}
    
    analysis = {}
    
    for metric in metrics_to_test:
        baseline_values = fold_results[baseline_key].get(metric, [])
        if not baseline_values:
            continue
        
        pairwise_results = []
        p_values_for_correction = []
        
        for condition, cond_data in fold_results.items():
            if condition == baseline_key:
                continue
            
            cond_values = cond_data.get(metric, [])
            if not cond_values or len(cond_values) != len(baseline_values):
                continue
            
            # Paired t-test
            t_stat, p_value = paired_ttest(cond_values, baseline_values)
            
            # Cohen's d
            d = cohens_d_paired(cond_values, baseline_values)
            
            # Delta CI
            ci_low, ci_high = delta_confidence_interval(cond_values, baseline_values)
            
            # Mean difference
            mean_diff = float(np.mean(np.array(cond_values) - np.array(baseline_values)))
            
            result = {
                'condition': condition,
                'baseline': baseline_key,
                'metric': metric,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': d,
                'mean_difference': mean_diff,
                'ci_95_low': ci_low,
                'ci_95_high': ci_high,
                'condition_mean': float(np.mean(cond_values)),
                'condition_std': float(np.std(cond_values)),
                'baseline_mean': float(np.mean(baseline_values)),
                'baseline_std': float(np.std(baseline_values)),
            }
            
            pairwise_results.append(result)
            p_values_for_correction.append((condition, p_value))
        
        # Holm-Bonferroni correction
        if p_values_for_correction:
            corrected = holm_bonferroni_correction(p_values_for_correction)
            correction_map = {c['comparison']: c for c in corrected}
            
            for result in pairwise_results:
                corr = correction_map.get(result['condition'], {})
                result['corrected_p'] = corr.get('corrected_p', result['p_value'])
                result['significant_corrected'] = corr.get('significant', False)
                result['significant_uncorrected'] = result['p_value'] < 0.05
        
        analysis[metric] = pairwise_results
    
    return analysis


def run_all_pairwise_comparisons(fold_results, metrics_to_test=None):
    """
    Run pairwise comparisons between ALL pairs of conditions.
    
    Returns dict organized by metric with all pairwise results.
    """
    if metrics_to_test is None:
        metrics_to_test = ['macro_f1', 'auc', 'balanced_accuracy']
    
    all_analysis = {}
    conditions = list(fold_results.keys())
    
    for metric in metrics_to_test:
        pairwise_results = []
        p_values_for_correction = []
        
        for c1, c2 in combinations(conditions, 2):
            vals1 = fold_results[c1].get(metric, [])
            vals2 = fold_results[c2].get(metric, [])
            
            if not vals1 or not vals2 or len(vals1) != len(vals2):
                continue
            
            t_stat, p_value = paired_ttest(vals1, vals2)
            d = cohens_d_paired(vals1, vals2)
            ci_low, ci_high = delta_confidence_interval(vals1, vals2)
            
            result = {
                'condition_1': c1,
                'condition_2': c2,
                'metric': metric,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': d,
                'ci_95_low': ci_low,
                'ci_95_high': ci_high,
                'mean_1': float(np.mean(vals1)),
                'mean_2': float(np.mean(vals2)),
            }
            
            pairwise_results.append(result)
            p_values_for_correction.append((f"{c1}_vs_{c2}", p_value))
        
        # Holm-Bonferroni correction
        if p_values_for_correction:
            corrected = holm_bonferroni_correction(p_values_for_correction)
            correction_map = {c['comparison']: c for c in corrected}
            
            for result in pairwise_results:
                key = f"{result['condition_1']}_vs_{result['condition_2']}"
                corr = correction_map.get(key, {})
                result['corrected_p'] = corr.get('corrected_p', result['p_value'])
                result['significant_corrected'] = corr.get('significant', False)
        
        all_analysis[metric] = pairwise_results
    
    return all_analysis


def save_statistical_analysis(analysis, config, dataset_name, task, label='vs_baseline'):
    """Save statistical analysis results."""
    base = config['data']['base_dir']
    stats_dir = os.path.join(base, config['outputs']['tables_dir'])
    os.makedirs(stats_dir, exist_ok=True)
    
    task_str = f"_{task}" if task else ""
    
    # Save as JSON
    json_file = os.path.join(stats_dir, 
                             f"{dataset_name}{task_str}_stats_{label}.json")
    with open(json_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Save as CSV tables per metric
    for metric, results in analysis.items():
        if results:
            df = pd.DataFrame(results)
            csv_file = os.path.join(stats_dir,
                                    f"{dataset_name}{task_str}_stats_{label}_{metric}.csv")
            df.to_csv(csv_file, index=False)
    
    logger.info(f"Statistical analysis saved: {json_file}")
    return json_file


def generate_latex_stats_table(analysis, metric='macro_f1'):
    """Generate a LaTeX-ready statistical comparison table."""
    if metric not in analysis:
        return ""
    
    results = analysis[metric]
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Statistical comparison for " + metric.replace('_', ' ') + r"}",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Condition & Mean & $\Delta$ & Cohen's d & p-value & Corrected p & Sig. \\",
        r"\hline",
    ]
    
    for r in results:
        sig = r"$\checkmark$" if r.get('significant_corrected', False) else ""
        lines.append(
            f"{r['condition']} & "
            f"{r['condition_mean']:.4f} & "
            f"{r['mean_difference']:+.4f} & "
            f"{r['cohens_d']:.3f} & "
            f"{r['p_value']:.4f} & "
            f"{r.get('corrected_p', r['p_value']):.4f} & "
            f"{sig} \\\\"
        )
    
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return '\n'.join(lines)
