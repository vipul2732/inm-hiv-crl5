#!/usr/bin/env python3
"""
Perform statistical tests to compare positive vs negative AlphaFold prediction sets.
Tests: Mann-Whitney U test, t-test, effect size (Cohen's d)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_effect_size(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def perform_statistical_tests(pos_data, neg_data, metric_name):
    """Perform statistical tests comparing positive vs negative sets."""
    pos_values = pos_data.dropna()
    neg_values = neg_data.dropna()
    
    if len(pos_values) == 0 or len(neg_values) == 0:
        return None
    
    results = {
        'metric': metric_name,
        'n_pos': len(pos_values),
        'n_neg': len(neg_values),
        'pos_mean': float(np.mean(pos_values)),
        'pos_median': float(np.median(pos_values)),
        'pos_std': float(np.std(pos_values, ddof=1)),
        'neg_mean': float(np.mean(neg_values)),
        'neg_median': float(np.median(neg_values)),
        'neg_std': float(np.std(neg_values, ddof=1)),
        'mean_diff': float(np.mean(pos_values) - np.mean(neg_values)),
        'median_diff': float(np.median(pos_values) - np.median(neg_values)),
    }
    
    # Mann-Whitney U test (non-parametric)
    try:
        u_stat, p_mannwhitney = stats.mannwhitneyu(pos_values, neg_values, alternative='two-sided')
        results['mannwhitney_u'] = float(u_stat)
        results['mannwhitney_p'] = float(p_mannwhitney)
        results['mannwhitney_significant'] = p_mannwhitney < 0.05
    except Exception as e:
        logger.warning(f"Mann-Whitney test failed for {metric_name}: {e}")
        results['mannwhitney_u'] = None
        results['mannwhitney_p'] = None
        results['mannwhitney_significant'] = None
    
    # Independent t-test (parametric)
    try:
        t_stat, p_ttest = stats.ttest_ind(pos_values, neg_values)
        results['ttest_t'] = float(t_stat)
        results['ttest_p'] = float(p_ttest)
        results['ttest_significant'] = p_ttest < 0.05
    except Exception as e:
        logger.warning(f"t-test failed for {metric_name}: {e}")
        results['ttest_t'] = None
        results['ttest_p'] = None
        results['ttest_significant'] = None
    
    # Effect size (Cohen's d)
    try:
        d = cohens_d(pos_values, neg_values)
        results['cohens_d'] = float(d)
        results['effect_size'] = interpret_effect_size(d)
    except Exception as e:
        logger.warning(f"Effect size calculation failed for {metric_name}: {e}")
        results['cohens_d'] = None
        results['effect_size'] = None
    
    return results


def analyze_condition(condition_name, pos_file, neg_file, output_dir):
    """Analyze a single condition."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing condition: {condition_name}")
    logger.info(f"{'='*80}")
    
    # Load data
    pos_df = pd.read_csv(pos_file, sep='\t')
    neg_df = pd.read_csv(neg_file, sep='\t')
    
    logger.info(f"Positive set: {len(pos_df)} pairs")
    logger.info(f"Negative set: {len(neg_df)} pairs")
    
    # Test metrics
    metrics = [
        ('ipTM', 'Interface TM-score'),
        ('mean_pLDDT', 'Mean pLDDT'),
        ('mean_interface_PAE', 'Mean Interface PAE'),
        ('frac_interface_PAE_lt10', 'Fraction PAE < 10'),
        ('frac_interface_PAE_lt15', 'Fraction PAE < 15'),
    ]
    
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"STATISTICAL TEST RESULTS: {condition_name}")
    print(f"{'='*80}\n")
    
    for metric_col, metric_name in metrics:
        if metric_col not in pos_df.columns or metric_col not in neg_df.columns:
            logger.warning(f"Metric {metric_col} not found in data")
            continue
        
        pos_data = pos_df[metric_col]
        neg_data = neg_df[metric_col]
        
        results = perform_statistical_tests(pos_data, neg_data, metric_name)
        if results is None:
            continue
        
        all_results.append(results)
        
        # Print results
        print(f"\n{metric_name} ({metric_col}):")
        print(f"  Positive: n={results['n_pos']}, mean={results['pos_mean']:.4f}, "
              f"median={results['pos_median']:.4f}, std={results['pos_std']:.4f}")
        print(f"  Negative: n={results['n_neg']}, mean={results['neg_mean']:.4f}, "
              f"median={results['neg_median']:.4f}, std={results['neg_std']:.4f}")
        print(f"  Difference: {results['mean_diff']:+.4f} (mean), {results['median_diff']:+.4f} (median)")
        
        if results['mannwhitney_p'] is not None:
            sig_str = "***" if results['mannwhitney_p'] < 0.001 else "**" if results['mannwhitney_p'] < 0.01 else "*" if results['mannwhitney_p'] < 0.05 else "ns"
            print(f"  Mann-Whitney U test: U={results['mannwhitney_u']:.2f}, p={results['mannwhitney_p']:.4e} {sig_str}")
        
        if results['ttest_p'] is not None:
            sig_str = "***" if results['ttest_p'] < 0.001 else "**" if results['ttest_p'] < 0.01 else "*" if results['ttest_p'] < 0.05 else "ns"
            print(f"  Independent t-test: t={results['ttest_t']:.3f}, p={results['ttest_p']:.4e} {sig_str}")
        
        if results['cohens_d'] is not None:
            print(f"  Effect size (Cohen's d): {results['cohens_d']:.3f} ({results['effect_size']})")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = output_dir / f'statistical_tests_{condition_name.lower()}.tsv'
        results_df.to_csv(output_file, sep='\t', index=False)
        logger.info(f"\nSaved results to {output_file}")
    
    return all_results


def main():
    output_dir = Path("Alphafold_multimer")
    output_dir.mkdir(exist_ok=True)
    
    conditions = [
        ('Wildtype', 
         output_dir / "colabfold_multimer_summary_wildtype_positive_set.tsv",
         output_dir / "colabfold_multimer_summary_wildtype_negative_set.tsv"),
        ('VIF',
         output_dir / "colabfold_multimer_summary_vif_positive_Set.tsv",
         output_dir / "colabfold_multimer_summary_vif_negative_set.tsv"),
        ('Mock',
         output_dir / "colabfold_multimer_summary_mock_positve_set.tsv",
         output_dir / "colabfold_multimer_summary_mock_negative_set.tsv"),
    ]
    
    all_condition_results = []
    
    for condition_name, pos_file, neg_file in conditions:
        if not pos_file.exists() or not neg_file.exists():
            logger.warning(f"Skipping {condition_name}: files not found")
            continue
        
        results = analyze_condition(condition_name, pos_file, neg_file, output_dir)
        if results:
            for r in results:
                r['condition'] = condition_name
            all_condition_results.extend(results)
    
    # Create combined summary
    if all_condition_results:
        combined_df = pd.DataFrame(all_condition_results)
        combined_file = output_dir / 'statistical_tests_combined.tsv'
        combined_df.to_csv(combined_file, sep='\t', index=False)
        logger.info(f"\nSaved combined results to {combined_file}")
        
        # Print summary for key metrics
        print(f"\n{'='*80}")
        print("SUMMARY: Key Metrics (ipTM and mean_pLDDT)")
        print(f"{'='*80}\n")
        
        key_metrics = ['Interface TM-score', 'Mean pLDDT']
        for metric in key_metrics:
            metric_data = combined_df[combined_df['metric'] == metric]
            if len(metric_data) > 0:
                print(f"\n{metric}:")
                for _, row in metric_data.iterrows():
                    sig_mw = "***" if row['mannwhitney_p'] < 0.001 else "**" if row['mannwhitney_p'] < 0.01 else "*" if row['mannwhitney_p'] < 0.05 else "ns" if row['mannwhitney_p'] is not None else "N/A"
                    print(f"  {row['condition']}: "
                          f"diff={row['mean_diff']:+.4f}, "
                          f"p={row['mannwhitney_p']:.4e} {sig_mw}, "
                          f"d={row['cohens_d']:.3f} ({row['effect_size']})")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
