"""
Phase 2: Anonymous Validation ("Back-Stage" Environment)
Script: 12_engineer_reddit_features.py

Description:
Transforms raw social longitudinal histories into normalized behavioral predictors 
(Night-Shifts, IAT, Volume). This guarantees methodological symmetry when 
comparing social telemetry against Version Control System (VCS) telemetry.
"""

import os
import numpy as np
import pandas as pd

INPUT_FILE = './data/raw/reddit_longitudinal_raw.parquet'
OUTPUT_FILE = './data/processed/reddit_engineered.parquet'

def prep_reddit_data():
    print("[*] Processing Reddit longitudinal histories...")
    if not os.path.exists(INPUT_FILE): return
        
    df = pd.read_parquet(INPUT_FILE)
    print("[*] Extracting temporal boundaries and stochastic intervals...")
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values(['author_id', 'date'])
    
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night_shift'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    
    df['iat_hours'] = df.groupby('author_id')['date'].diff().dt.total_seconds().div(3600).fillna(0)
    
    print("[*] Normalizing distributions for Deep Learning stability...")
    df['norm_iat'] = np.clip(df['iat_hours'], 0, 720) / 720.0
    df['norm_msg_len'] = np.clip(df['text_length'], 0, 200) / 200.0 
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"[*] Success! Engineered metadata saved to: {OUTPUT_FILE}") # [cite: 963]

if __name__ == "__main__":
    prep_reddit_data()