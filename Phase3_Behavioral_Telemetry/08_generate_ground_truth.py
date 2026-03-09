"""
Phase 3: Behavioral Telemetry in Version Control (GitHub)
Script: 08_generate_ground_truth.py

Description:
Applies unsupervised learning (K-Means) over the engineered behavioral metrics 
to establish the Ground Truth. Since VCS platforms lack clinical psychometric labels, 
this algorithm isolates the anomalous cluster (characterized by high operational friction 
and severe night-shifts) representing the 11.8% of developers experiencing Burnout.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================================
# CLUSTERING CONFIGURATION (Section 3.5.3)
# =====================================================================
INPUT_FILE = './data/processed/github_mass_engineered.parquet'
OUTPUT_DIR = './results/clusters'
N_CLUSTERS = 2
RANDOM_SEED = 42

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def discover_burnout_clusters():
    print("[*] Initializing Unsupervised Learning (K-Means) Pipeline...")
    df = pd.read_parquet(INPUT_FILE).fillna(0)
    print(f"[*] Total telemetry records loaded: {len(df)}")
    
    # Behavioral features ignoring textual semantics (MSR Paradigm)
    features = ['is_night_shift', 'is_weekend', 'deletion_ratio', 'norm_iat']
    X = df[features].copy()
    X['short_message_flag'] = 1.0 - df['norm_msg_len'] 
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"[*] Fitting K-Means (k={N_CLUSTERS}) to detect chronic stress signatures...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    df['label'] = kmeans.fit_predict(X_scaled)
    
    # Automated cluster mapping: The cluster with higher night-shifts is mapped to Burnout (1)
    if df.groupby('label')['is_night_shift'].mean()[0] > df.groupby('label')['is_night_shift'].mean()[1]:
        df['label'] = 1 - df['label']
        
    cluster_dist = df['label'].value_counts(normalize=True) * 100
    print("\n[*] GROUND TRUTH DISTRIBUTION DISCOVERED:")
    print(f"    -> Healthy Workflow (0): {cluster_dist[0]:.1f}%")
    print(f"    -> Burnout / Stress (1): {cluster_dist[1]:.1f}%")
    
    # Save the labeled dataset for supervised learning (Oracle)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_parquet('./data/processed/github_labeled_ground_truth.parquet', index=False)
    print("[*] Success! Labeled dataset saved for Phase 4.")

if __name__ == "__main__":

    discover_burnout_clusters()
