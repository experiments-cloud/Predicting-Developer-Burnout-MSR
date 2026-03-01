"""
Phase 1: Semantic Blindness in Formal Environments (Stack Overflow)
Script: 02_generate_churn_labels.py

Description:
Establishes the Ground Truth for formal environments. Since technical platforms 
lack clinical labels, this script operationalizes "Burnout" as platform abandonment 
(Churn), defined mathematically as continuous inactivity exceeding 180 days after 
a consistent contribution history (Section 3.3).
"""

import os
import gc
import numpy as np
import pandas as pd

# =====================================================================
# OPERATIONAL DEFINITION HYPERPARAMETERS (Section 3.3)
# =====================================================================
INPUT_FILE = './data/processed/dataset_anonymized.csv'
OUTPUT_TRAIN = './data/processed/train_data.parquet'
OUTPUT_TEST = './data/processed/test_data.parquet'

MIN_POSTS = 5           # Removes 'tourist' users without a track record
CHURN_DAYS = 180        # Threshold for operational burnout
TEST_SIZE = 0.2         # 20% hold-out for validation

def generate_operational_labels():
    print("[*] Stage 2: Processing high-volume dataset for label generation...")
    
    # Pass 1: Scan metadata to calculate trajectories
    cols_needed = ['user_hash', 'date']
    df_meta = pd.read_csv(INPUT_FILE, usecols=cols_needed, parse_dates=['date'])
    
    max_date = df_meta['date'].max()
    print(f"[*] Study cutoff date detected: {max_date.date()}")
    
    user_stats = df_meta.groupby('user_hash')['date'].agg(['max', 'count'])
    user_stats.columns = ['last_seen', 'post_count']
    
    valid_users = user_stats[user_stats['post_count'] >= MIN_POSTS].copy()
    print(f"[*] Valid developers (>= {MIN_POSTS} interactions): {len(valid_users)}")
    
    # Establish Churn/Burnout definition
    cutoff_date = max_date - pd.Timedelta(days=CHURN_DAYS)
    valid_users['label'] = (valid_users['last_seen'] < cutoff_date).astype(int)
    
    burnout_rate = valid_users['label'].mean()
    print(f"[*] Operational Burnout (Churn) rate: {burnout_rate:.2%}")
    
    # Strict Train/Test split by user to prevent data leakage
    all_user_ids = valid_users.index.values
    np.random.seed(42)
    np.random.shuffle(all_user_ids)
    
    split_idx = int(len(all_user_ids) * (1 - TEST_SIZE))
    train_users_set = set(all_user_ids[:split_idx])
    
    user_map = valid_users['label'].to_dict()
    
    del df_meta, user_stats, valid_users, all_user_ids
    gc.collect()
    
    # Pass 2: Chunk processing to apply labels to the full dataset
    print("[*] Generating final datasets (Chunking applied for memory safety)...")
    
    # Ensure fastparquet or pyarrow is available for chunked appending
    try:
        chunksize = 500000
        processed_rows = 0
        
        for chunk in pd.read_csv(INPUT_FILE, chunksize=chunksize, dtype={'user_hash': 'str'}):
            chunk['label'] = chunk['user_hash'].map(user_map)
            chunk = chunk.dropna(subset=['label'])
            chunk['label'] = chunk['label'].astype(int) 
            
            if chunk.empty: continue
                
            is_train = chunk['user_hash'].isin(train_users_set)
            train_chunk = chunk[is_train]
            test_chunk = chunk[~is_train]
            
            append_train = os.path.exists(OUTPUT_TRAIN)
            append_test = os.path.exists(OUTPUT_TEST)
            
            if not train_chunk.empty:
                train_chunk.to_parquet(OUTPUT_TRAIN, engine='fastparquet', index=False, append=append_train)
            if not test_chunk.empty:
                test_chunk.to_parquet(OUTPUT_TEST, engine='fastparquet', index=False, append=append_test)
                
            processed_rows += len(chunk)
            print(f"    -> Processed {processed_rows} valid interactions...", end='\r')

        print("\n[*] Success! Labeling and data splitting complete.")
    except Exception as e:
        print(f"\n[!] ERROR during Parquet creation: Ensure 'fastparquet' is installed. {e}")

if __name__ == "__main__":
    generate_operational_labels()