"""
Phase 1: Semantic Blindness in Formal Environments (Stack Overflow)
Script: 03_apply_downsampling.py

Description:
Applies a strict 50/50 downsampling strategy to the training dataset.
This forces perfect class balance between active and churned developers, ensuring 
that the NLP models do not converge into local optima by predicting the majority 
class (Section 3.3).
"""

import gc
import pandas as pd

# =====================================================================
# SAMPLING CONFIGURATION (Section 3.3)
# =====================================================================
INPUT_TRAIN = './data/processed/train_data.parquet'
OUTPUT_BALANCED = './data/processed/train_balanced.parquet'
RANDOM_SEED = 42

def apply_strict_downsampling():
    print("[*] Initializing strict class balancing (50/50 strategy)...")
    
    # Read only metadata to optimize RAM
    df_meta = pd.read_parquet(INPUT_TRAIN, columns=['user_hash', 'label'])
    users_unique = df_meta.drop_duplicates(subset=['user_hash'])
    
    active_users = users_unique[users_unique['label'] == 0]
    burnout_users = users_unique[users_unique['label'] == 1]
    
    n_active = len(active_users)
    n_burnout = len(burnout_users)
    
    print("[*] Initial Class Distribution:")
    print(f"    -> Active (Class 0): {n_active} developers")
    print(f"    -> Burnout (Class 1): {n_burnout} developers")
    
    # Downsample the majority class to match the active user count
    print(f"[*] Downsampling Burnout class to {n_active} developers...")
    burnout_sample = burnout_users.sample(n=n_active, random_state=RANDOM_SEED)
    
    # Authorized user set for the final balanced dataset
    keep_users = set(active_users['user_hash']).union(set(burnout_sample['user_hash']))
    print(f"[*] Total developers for balanced training: {len(keep_users)}")
    
    del df_meta, users_unique, active_users, burnout_users, burnout_sample
    gc.collect()
    
    # Re-read full dataset and apply the filter
    print("[*] Generating definitive balanced parquet file...")
    df_full = pd.read_parquet(INPUT_TRAIN)
    df_balanced = df_full[df_full['user_hash'].isin(keep_users)]
    
    df_balanced.to_parquet(OUTPUT_BALANCED, index=False)
    
    print(f"[*] Success! Balanced dataset saved to: {OUTPUT_BALANCED}")
    print(f"[*] Final label distribution forced to:\n{df_balanced['label'].value_counts(normalize=True)}")

if __name__ == "__main__":
    apply_strict_downsampling()