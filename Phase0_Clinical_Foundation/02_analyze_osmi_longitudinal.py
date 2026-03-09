"""
Phase 0: Clinical Foundation & Problem Statement
Script: 02_analyze_osmi_longitudinal.py

Description:
Applies a fuzzy-matching heuristic to unify shifting schema structures across 
10 years of OSMI surveys (2014-2023). It longitudinally proves that Professional 
Masking is a systemic, enduring phenomenon in the tech industry, validating the 
methodological need to abandon text-based NLP analysis in formal environments.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re

DATA_FOLDER = './data/raw/osmi/'
PLOT_DIR = './plots'

def find_treatment_col(columns):
    for col in columns:
        c_lower = col.lower()
        if c_lower == 'treatment' or ('sought treatment' in c_lower and 'mental health' in c_lower): return col
    return None

def find_consequence_col(columns):
    for col in columns:
        c_lower = col.lower()
        if c_lower == 'mental_health_consequence' or ('discussing' in c_lower and 'negative consequence' in c_lower and 'employer' in c_lower): return col
    return None

def analyze_longitudinal_osmi():
    print("[*] Initializing DEEP Historical Scan of OSMI Surveys (2014-2023)...")
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    if not all_files: return

    historical_data = []

    for file in all_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            match = re.search(r'\d{4}', os.path.basename(file))
            year = match.group(0) if match else "2014"
            
            col_treatment = find_treatment_col(df.columns)
            col_consequence = find_consequence_col(df.columns)
            
            if col_treatment and col_consequence:
                treat_data = df[col_treatment].astype(str).str.lower()
                cons_data = df[col_consequence].astype(str).str.lower()
                
                treat_yes = treat_data.str.contains('yes|1|true').sum()
                treat_total = len(treat_data[treat_data != 'nan'])
                
                masking_yes = cons_data.str.contains('yes|maybe|some of them').sum()
                masking_total = len(cons_data[cons_data != 'nan'])
                
                if treat_total > 0 and masking_total > 0:
                    historical_data.append({
                        'Year': int(year),
                        'Sought_Treatment_Pct': (treat_yes / treat_total) * 100,
                        'Fear_of_Consequences_Pct': (masking_yes / masking_total) * 100,
                        'Sample_Size': masking_total
                    })
        except Exception as e:
            pass

    trend_df = pd.DataFrame(historical_data).groupby('Year').mean().reset_index().sort_values('Year')

    print("\n[*] Rendering Decade-long Evolution Artifact...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    plt.plot(trend_df['Year'], trend_df['Fear_of_Consequences_Pct'], 
             marker='o', color='firebrick', linewidth=3, markersize=8, 
             label='Fear of Negative Consequences (Masking)')
             
    plt.plot(trend_df['Year'], trend_df['Sought_Treatment_Pct'], 
             marker='s', color='navy', linewidth=3, markersize=8, linestyle='--',
             label='Sought Professional Treatment')
    
    plt.title('Evolution of "Professional Masking" in the Tech Industry (2014-2023)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('OSMI Survey Year', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage of Developers (%)', fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    plt.xticks(trend_df['Year'])
    plt.legend(loc="center right", fontsize=11)
    
    plt.fill_between(trend_df['Year'], trend_df['Sought_Treatment_Pct'], trend_df['Fear_of_Consequences_Pct'], color='gray', alpha=0.1)
    
    avg_masking = trend_df['Fear_of_Consequences_Pct'].mean()
    avg_treatment = trend_df['Sought_Treatment_Pct'].mean()
    print(f"\n[*] TEXTBOOK METRICS FOR SECTION 4.1:")
    print(f"    -> Decade Average Fear of Consequences (Masking): {avg_masking:.1f}%")
    print(f"    -> Decade Average Sought Treatment: {avg_treatment:.1f}%")
    
    plt.tight_layout()
    output_path = f'{PLOT_DIR}/osmi_historical_trend_en.png'
    plt.savefig(output_path, dpi=300)
    print(f"[*] Success! English-localized artifact saved to: {output_path}")

if __name__ == "__main__":

    analyze_longitudinal_osmi()
