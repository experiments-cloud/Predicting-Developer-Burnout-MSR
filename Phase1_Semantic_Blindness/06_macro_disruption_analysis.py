"""
Phase 1: Semantic Blindness in Formal Environments (Stack Overflow)
Script: 06_macro_disruption_analysis.py

Description:
Analyzes the longitudinal structural integrity of the Stack Overflow dataset.
It correlates the structural drop in platform participation with the launch 
of Generative AI (Nov 2022). This script justifies the methodological transition 
towards GitHub Version Control Telemetry, as traditional Q&A forums are 
experiencing a severe volume and participation crisis (Section 4.1).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# MACROECONOMIC ANALYSIS CONFIGURATION
# =====================================================================
INPUT_FILE = './data/processed/train_data.parquet' 
OUTPUT_DIR = './results/macro_analysis'

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)

def run_macro_analysis():
    print("[*] Loading massive historical dataset (Metadata only)...")
    df = pd.read_parquet(INPUT_FILE, columns=['date', 'user_hash', 'code_ratio'])
    
    print("[*] Processing longitudinal timeline...")
    df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True).dt.tz_localize(None)
    df['year_month'] = df['date'].dt.to_period('M')
    
    print("[*] Aggregating monthly participation statistics...")
    monthly_stats = df.groupby('year_month').agg(
        total_posts=('user_hash', 'count'),
        unique_users=('user_hash', 'nunique'),
        avg_code_ratio=('code_ratio', 'mean')
    ).reset_index()
    
    monthly_stats['date_plot'] = monthly_stats['year_month'].dt.to_timestamp()
    
    # Generative AI Disruption Marker
    disruption_date = pd.to_datetime('2022-11-01')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("[*] Generating Artifact: Structural Drop in Participation...")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color_posts = '#1f77b4' 
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Total Monthly Posts', color=color_posts, fontweight='bold')
    ax1.plot(monthly_stats['date_plot'], monthly_stats['total_posts'], color=color_posts, linewidth=2.5, label='Post Volume')
    ax1.tick_params(axis='y', labelcolor=color_posts)
    
    ax1.axvline(x=disruption_date, color='#d62728', linestyle='--', linewidth=2, label='Generative AI Launch (Nov 2022)')
    
    ax2 = ax1.twinx()  
    color_users = '#ff7f0e' 
    ax2.set_ylabel('Unique Active Users', color=color_users, fontweight='bold')  
    ax2.plot(monthly_stats['date_plot'], monthly_stats['unique_users'], color=color_users, linewidth=2, linestyle=':', label='Active Users')
    ax2.tick_params(axis='y', labelcolor=color_users)
    
    fig.suptitle('Structural Drop in Stack Overflow Participation', fontsize=16, fontweight='bold')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    fig.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_macro_volume_disruption.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[*] Generating Artifact: Code Composition Evolution...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_stats, x='date_plot', y='avg_code_ratio', color='#2ca02c', linewidth=2.5)
    plt.axvline(x=disruption_date, color='#d62728', linestyle='--', linewidth=2, label='Generative AI Launch')
    
    plt.title('Text Composition Evolution (Code Ratio)', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontweight='bold')
    plt.ylabel('Average Code Ratio (0 to 1)', fontweight='bold')
    plt.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_macro_code_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] Success! Macro-environmental artifacts generated at: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_macro_analysis()