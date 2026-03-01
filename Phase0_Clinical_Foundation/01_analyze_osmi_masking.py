"""
Phase 0: Clinical Foundation & Problem Statement
Script: 01_analyze_osmi_masking.py

Description:
Analyzes the 'Mental Health in Tech Survey' (OSMI) to provide clinical 
and statistical evidence of "Professional Masking". It demonstrates the severe 
discrepancy between the actual prevalence of mental exhaustion (treatment seeking) 
and the fear of discussing it in formal work environments due to negative consequences.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_FILE = './data/raw/osmi_survey.csv'
PLOT_DIR = './plots'

def analyze_osmi_masking():
    print("[*] Initializing OSMI Clinical Survey Analysis (Mental Health in Tech)...")
    if not os.path.exists(INPUT_FILE): return

    df = pd.read_csv(INPUT_FILE)
    tech_df = df[df['tech_company'] == 'Yes'].copy()
    print(f"[*] Total Software Engineering Professionals Analyzed: {len(tech_df)}")

    # Metric 1: Fear of Consequences (The root of Professional Masking)
    consequences = tech_df['mental_health_consequence'].value_counts(normalize=True) * 100
    
    # Metric 2: Willingness to discuss with peers (Front-Stage vs Back-Stage)
    coworkers = tech_df['coworkers'].value_counts(normalize=True) * 100

    print("[*] Rendering Evidence Artifact for Professional Masking...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Negative consequences
    sns.barplot(x=consequences.index, y=consequences.values, ax=axes[0], palette="Reds_r")
    axes[0].set_title('Do you think discussing a mental health issue\nwith your employer would have negative consequences?', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Percentage of Developers (%)', fontsize=12)
    axes[0].set_ylim(0, 100)
    
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Subplot 2: Coworker discussion
    sns.barplot(x=coworkers.index, y=coworkers.values, ax=axes[1], palette="Blues_r", order=['No', 'Some of them', 'Yes'])
    axes[1].set_title('Would you be willing to discuss a mental\nhealth issue with your coworkers?', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Percentage of Developers (%)', fontsize=12)
    axes[1].set_ylim(0, 100)

    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Clinical Evidence of "Professional Masking" in Software Engineering', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    output_path = f'{PLOT_DIR}/osmi_professional_masking_en.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[*] Success! Artifact saved to: {output_path}")

if __name__ == "__main__":
    analyze_osmi_masking()