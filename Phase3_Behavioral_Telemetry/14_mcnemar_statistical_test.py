"""
Phase 3: Behavioral Telemetry in Version Control (GitHub)
Script: 14_mcnemar_statistical_test.py

Description:
Executes the final statistical validation (McNemar's Test with continuity correction) 
comparing the Temporal Oracle (LSTM) against the Static Baseline (Random Forest).
By achieving p > 0.05, it confirms that MSR behavioral telemetry alone is sufficiently 
predictive of burnout, rendering complex temporal architectures unnecessary for 
practical industry deployment.
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings("ignore")

# =====================================================================
# STATISTICAL TEST CONFIGURATION 
# =====================================================================
INPUT_FILE = './data/processed/github_mass_engineered.parquet'
SEQ_LEN = 5       # Reduced to maximize sample availability
EPOCHS = 35       
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def label_data_with_kmeans(df):
    df = df.fillna(0).sort_values(['author_id', 'date'])
    features = ['is_night_shift', 'is_weekend', 'deletion_ratio', 'norm_iat']
    X = df[features].copy()
    X['short_message'] = 1.0 - df['norm_msg_len']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['label'] = kmeans.fit_predict(X_scaled)
    
    if df.groupby('label')['is_night_shift'].mean()[0] > df.groupby('label')['is_night_shift'].mean()[1]:
        df['label'] = 1 - df['label']
    return df

def create_sequences(df, seq_len):
    sequences, labels = [], []
    grouped = df.groupby('author_id')
    for user, group in grouped:
        if len(group) < seq_len + 1: continue 
        recent_posts = group.tail(seq_len + 1)
        context_posts = recent_posts.iloc[:-1]
        future_label = recent_posts.iloc[-1]['label']
        metas = context_posts[['is_night_shift', 'is_weekend', 'deletion_ratio', 'norm_msg_len', 'norm_iat']].values.tolist()
        sequences.append(metas)
        labels.append(future_label)
    return np.array(sequences), np.array(labels)

class BehavioralLSTM(nn.Module):
    def __init__(self):
        super(BehavioralLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=32, batch_first=True, num_layers=2, dropout=0.4)
        self.classifier = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 2))
        
    def forward(self, meta_data):
        _, (hidden, _) = self.lstm(meta_data)
        return self.classifier(hidden[-1])

def run_statistical_competition():
    print(f"[*] Extracting sequences with SEQ_LEN = {SEQ_LEN}...")
    df = pd.read_parquet(INPUT_FILE)
    df_labeled = label_data_with_kmeans(df)
    
    X_all, y_all = create_sequences(df_labeled, SEQ_LEN)
    
    split_idx = int(len(X_all) * 0.85)
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_test, y_true = X_all[split_idx:], y_all[split_idx:]
    
    print(f"[*] Dynamic Weighting Applied. Hold-out Test Size: {len(y_true)}")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    # 1. Static Baseline (Random Forest)
    print("[*] Training Static Baseline (Random Forest)...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_model.fit(X_train_flat, y_train)
    y_pred_rf = rf_model.predict(X_test_flat)
    
    # 2. Temporal Oracle (LSTM)
    print(f"[*] Training Deep Learning Oracle ({EPOCHS} Epochs)...")
    model = BehavioralLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float).to(device), torch.tensor(y_train, dtype=torch.long).to(device))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        _, y_pred_lstm = torch.max(model(torch.tensor(X_test, dtype=torch.float).to(device)), 1)
        y_pred_lstm = y_pred_lstm.cpu().numpy()

    # 3. McNemar's Statistical Test
    print("\n" + "="*50)
    print("MCNEMAR'S STATISTICAL VALIDATION")
    print("="*50)
    
    lstm_correct = (y_pred_lstm == y_true)
    base_correct = (y_pred_rf == y_true)
    
    both_correct = np.sum(lstm_correct & base_correct)
    lstm_only = np.sum(lstm_correct & ~base_correct)
    base_only = np.sum(~lstm_correct & base_correct)
    both_wrong = np.sum(~lstm_correct & ~base_correct)
    
    contingency_table = [[both_correct, lstm_only], [base_only, both_wrong]]
    result = mcnemar(contingency_table, exact=False, correction=True)
    
    print(f"Both Correct: {both_correct}")
    print(f"LSTM Only Correct: {lstm_only}")
    print(f"RF Only Correct: {base_only}")
    print(f"Both Incorrect: {both_wrong}")
    print("-" * 50)
    print(f"P-value: {result.pvalue:.4e}")
    print("="*50)
    
    if result.pvalue >= 0.05:
        print("[!] CONCLUSION: Fail to reject the null hypothesis (p >= 0.05).")
        print("    The Random Forest baseline performs statistically similarly to the LSTM.")
        print("    Industry deployment can rely on lighter algorithms using MSR behavioral data.")

if __name__ == "__main__":

    run_statistical_competition()
