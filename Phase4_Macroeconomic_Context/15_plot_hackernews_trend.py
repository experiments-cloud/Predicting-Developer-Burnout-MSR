"""
Phase 4: Macroeconomic Context & Ecological Validity
Script: 15_plot_hackernews_trend.py

Description:
Correlates the volume of burnout discussions on Hacker News with critical 
industry milestones. By applying a 3-month moving average smoothing, it 
visually demonstrates how global events like the COVID-19 Shift (2020) and 
the Big Tech Layoffs (2022) exacerbate developer stress levels.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = './data/raw/hackernews_mass_raw.parquet'
PLOT_DIR = './plots'

def plot_historical_trend():
    print("[*] Loading Hacker News longitudinal dataset...")
    df = pd.read_parquet(INPUT_FILE)
    
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.set_index('created_at', inplace=True)
    
    # Resample monthly and apply a 3-month rolling average for academic plotting
    monthly_trend = df.resample('ME').size().reset_index(name='discussions')
    monthly_trend['smoothed'] = monthly_trend['discussions'].rolling(window=3, center=True).mean()

    print("[*] Rendering Macroeconomic Trend Plot (2019 - 2026)...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    plt.plot(monthly_trend['created_at'], monthly_trend['discussions'], alpha=0.3, color='gray', label='Raw Monthly Volume')
    plt.plot(monthly_trend['created_at'], monthly_trend['smoothed'], color='firebrick', linewidth=2.5, label='Smoothed Trend (3-Month MA)')
    
    # Milestone 1: COVID-19 Pandemic (Remote Work Shift)
    pandemic_date = pd.to_datetime('2020-03-01')
    plt.axvline(pandemic_date, color='black', linestyle='--', alpha=0.6)
    plt.text(pd.to_datetime('2020-03-15'), plt.ylim()[1]*0.85, 'Pandemic Shift\n(Remote Work)', fontsize=11, fontweight='bold')
    
    # Milestone 2: Generative AI / Big Tech Layoffs
    layoffs_date = pd.to_datetime('2022-11-01')
    plt.axvline(layoffs_date, color='black', linestyle='--', alpha=0.6)
    plt.text(pd.to_datetime('2022-11-15'), plt.ylim()[1]*0.85, 'Big Tech Crisis\n(Mass Layoffs)', fontsize=11, fontweight='bold')
    
    plt.title('Historical Evolution of Burnout Discussions on Hacker News (2019-2026)', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Mentions of "Burnout"', fontsize=12, fontweight='bold')
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    
    output_path = f'{PLOT_DIR}/hackernews_burnout_trend_en.png'
    plt.savefig(output_path, dpi=300)
    print(f"[*] Success! English-localized plot saved at: {output_path}")

if __name__ == "__main__":
    plot_historical_trend()