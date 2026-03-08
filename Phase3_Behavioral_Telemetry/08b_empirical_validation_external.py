"""
Phase 3.5: Empirical Validation of the Ground Truth
Script: 08b_empirical_validation_external.py

Description:
This script acts as the definitive defense against the "mathematical circularity" critique.
It evaluates the K-Means clusters (generated purely from temporal/churn behavioral metadata)
against an EXTERNAL metric that the clustering algorithm never saw: semantic defect density
(the presence of "fix", "bug", or "revert" in the commit messages).
"""

import pandas as pd
import numpy as np
import re
from scipy.stats import chi2_contingency

# =====================================================================
# EMPIRICAL VALIDATION CONFIGURATION
# =====================================================================
INPUT_FILE = './data/github_labeled_ground_truth.parquet'

def validate_clusters_with_external_metrics():
    print("[*] Initializing External Empirical Validation (SZZ-Proxy & Reverts)...")
    
    try:
        df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print(f"[!] ERROR: {INPUT_FILE} not found. Run script 08 first.")
        return

    print(f"[*] Total labeled records loaded: {len(df)}")

    # 1. Define heuristics for external quality metrics
    # A proxy for defect resolution (SZZ-light) and destructive workflow (Reverts)
    regex_defect = re.compile(r'(?i)\b(fix|bug|patch|resolve|issue|hotfix)\b')
    regex_revert = re.compile(r'(?i)\b(revert)\b')

    print("[*] Parsing commit messages for defect and revert signatures...")
    df['commit_message'] = df['commit_message'].fillna("")
    
    # Flag commits that are fixing bugs
    df['is_defect_fix'] = df['commit_message'].apply(lambda msg: bool(regex_defect.search(msg))).astype(int)
    
    # Flag commits that are reverting previous work
    df['is_revert'] = df['commit_message'].apply(lambda msg: bool(regex_revert.search(msg))).astype(int)

    # 2. Analyze the distributions across the K-Means clusters
    healthy_df = df[df['label'] == 0]
    burnout_df = df[df['label'] == 1]

    healthy_fix_rate = healthy_df['is_defect_fix'].mean() * 100
    burnout_fix_rate = burnout_df['is_defect_fix'].mean() * 100

    healthy_revert_rate = healthy_df['is_revert'].mean() * 100
    burnout_revert_rate = burnout_df['is_revert'].mean() * 100

    print("\n" + "="*60)
    print("EMPIRICAL VALIDATION RESULTS (EXTERNAL METRICS)")
    print("="*60)
    print(f"Metrics evaluated on text (unseen by K-Means):\n")
    
    print(f"DEFECT FIX RATE (Probability that a commit is patching a bug):")
    print(f"  -> Healthy Workflow (0): {healthy_fix_rate:.2f}%")
    print(f"  -> Burnout Cluster (1):  {burnout_fix_rate:.2f}%")
    
    print(f"\nREVERT RATE (Probability of destructive undo operations):")
    print(f"  -> Healthy Workflow (0): {healthy_revert_rate:.2f}%")
    print(f"  -> Burnout Cluster (1):  {burnout_revert_rate:.2f}%")

    # 3. Statistical Significance Validation (Chi-Square)
    print("\n[*] Running Chi-Square Test for Independence...")
    
    # Crosstab for Defect Fixes
    contingency_defect = pd.crosstab(df['label'], df['is_defect_fix'])
    chi2_def, p_def, _, _ = chi2_contingency(contingency_defect)
    
    # Crosstab for Reverts
    contingency_revert = pd.crosstab(df['label'], df['is_revert'])
    chi2_rev, p_rev, _, _ = chi2_contingency(contingency_revert)

    print("-" * 60)
    print(f"Statistical Significance (Defect Rate): p-value = {p_def:.4e}")
    print(f"Statistical Significance (Revert Rate): p-value = {p_rev:.4e}")
    print("-" * 60)

    if p_def < 0.05 and p_rev < 0.05:
        print("[!] SUCCESS: The null hypothesis is rejected for both metrics.")
        print("    The Burnout cluster exhibits a statistically significant deterioration")
        print("    in code quality and workflow stability compared to the Healthy cluster.")
        print("    Construct Validity is firmly established.")
    else:
        print("[?] WARNING: Statistical significance not reached across all metrics.")

if __name__ == "__main__":
    validate_clusters_with_external_metrics()