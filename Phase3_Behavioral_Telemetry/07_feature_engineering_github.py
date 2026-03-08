"""
Phase 3: Behavioral Telemetry in Version Control (GitHub)
Script: 07_feature_engineering_github.py

Description:
Transforms raw Version Control System (VCS) commits into actionable behavioral metrics.
It engineers the precise predictors defined in Section 3.5.2: Boundary Dissolution 
(Night-Shifts, Weekend work) and Operational Friction (Deletion Ratio, IAT).
This pipeline processes the massive 119k+ record dataset required for the LSTM Oracle.
"""

import os
import numpy as np
import pandas as pd

# =====================================================================
# FEATURE ENGINEERING HYPERPARAMETERS (Section 3.5.2)
# =====================================================================
INPUT_GITHUB = './data/raw/github_real_mass.parquet' 
OUTPUT_CLEAN = './data/processed/github_mass_engineered.parquet'

def engineer_behavioral_features():
    print(f"[*] Initializing Behavioral Feature Engineering for MSR...")
    if not os.path.exists(INPUT_GITHUB):
        print(f"[!] ERROR: Input file {INPUT_GITHUB} not found.")
        return

    df = pd.read_parquet(INPUT_GITHUB)
    print(f"[*] Raw telemetry records loaded: {len(df)}")
    
    print("[*] Filtering out automated bots and CI/CD runners...")
    df = df[~df['author_id'].str.contains('bot|actions|dependabot|renovate', case=False, na=False)]
    print(f"[*] Human telemetry records remaining: {len(df)}")
    
    print("[*] Extracting Boundary Dissolution features (Temporal)...")
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values(['author_id', 'date'])
    
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Night-shift defined clinically as extreme operational hours (00:00 - 05:00)
    df['is_night_shift'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    
    # Inter-Arrival Time (IAT) measures erratic pause patterns
    df['iat_hours'] = df.groupby('author_id')['date'].diff().dt.total_seconds().div(3600).fillna(0)
    
    print("[*] Extracting Operational Friction features (Code Churn)...")
    df['total_churn'] = df['lines_added'] + df['lines_deleted']
    
    # Deletion Ratio isolates destructive refactoring from standard development
    df['deletion_ratio'] = df['lines_deleted'] / (df['total_churn'] + 1)
    
    df['commit_message'] = df['commit_message'].fillna("")
    df['message_length'] = df['commit_message'].apply(lambda x: len(str(x).split()))
    
    print("[*] Applying strict normalization bounds for Deep Learning stability...")
    df['norm_churn'] = np.clip(df['total_churn'], 0, 1000) / 1000.0
    df['norm_iat'] = np.clip(df['iat_hours'], 0, 720) / 720.0 
    df['norm_msg_len'] = np.clip(df['message_length'], 0, 50) / 50.0
    
    os.makedirs(os.path.dirname(OUTPUT_CLEAN), exist_ok=True)
    df.to_parquet(OUTPUT_CLEAN, index=False)
    print(f"[*] Success! Engineered behavioral dataset saved to: {OUTPUT_CLEAN}")

if __name__ == "__main__":

    engineer_behavioral_features()

