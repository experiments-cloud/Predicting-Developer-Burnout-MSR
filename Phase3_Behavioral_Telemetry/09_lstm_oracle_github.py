"""
Phase 3: Behavioral Telemetry in Version Control (GitHub)
Script: 09_lstm_oracle_github.py

Description:
Trains the Deep Learning Temporal Oracle. It processes sliding windows of 
consecutive VCS interactions to predict an imminent operational collapse.
This script demonstrates that physical behavioral telemetry is highly predictive 
of developer burnout, breaking the semantic stagnation observed in formal forums.
"""

import warnings
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore") 

# =====================================================================
# LSTM ORACLE HYPERPARAMETERS (Section 3.6.1 & 3.6.2)
# =====================================================================
INPUT_FILE = './data/processed/github_labeled_ground_truth.parquet'
SEQ_LEN = 5          
BATCH_SIZE = 64      
EPOCHS = 15          
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_oracle_sequences(df, seq_len):
    """Generates sliding windows ensuring strict temporal causality (No Data Leakage)."""
    print(f"[*] Building predictive timeline sequences (SEQ_LEN = {seq_len})...")
    sequences, labels = [], []
    grouped = df.groupby('author_id')
    
    for user, group in grouped:
        if len(group) < seq_len + 1: continue 
            
        recent_posts = group.tail(seq_len + 1)
        context_posts = recent_posts.iloc[:-1] # The context
        future_label = recent_posts.iloc[-1]['label'] # The future target
        
        metas = context_posts[['is_night_shift', 'is_weekend', 'deletion_ratio', 'norm_msg_len', 'norm_iat']].values.tolist()
        sequences.append(metas)
        labels.append(future_label)
        
    return sequences, labels

class MassGithubDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self): return len(self.labels)
    def __getitem__(self, index):
        return {
            'meta': torch.tensor(self.sequences[index], dtype=torch.float),
            'targets': torch.tensor(int(self.labels[index]), dtype=torch.long)
        }

class OracleLSTM(nn.Module):
    def __init__(self):
        super(OracleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=5, hidden_size=32, batch_first=True, num_layers=2, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 2)
        )
        
    def forward(self, meta_data):
        _, (hidden, _) = self.lstm(meta_data)
        return self.classifier(hidden[-1])

def train_oracle():
    df = pd.read_parquet(INPUT_FILE)
    sequences, labels = create_oracle_sequences(df, SEQ_LEN)
    print(f"[*] Total valid developer trajectories extracted: {len(sequences)}")
    
    split_idx = int(len(sequences) * 0.85) 
    train_loader = DataLoader(MassGithubDataset(sequences[:split_idx], labels[:split_idx]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MassGithubDataset(sequences[split_idx:], labels[split_idx:]), batch_size=BATCH_SIZE, shuffle=False)
    
    # Dynamic class weights to penalize minority class misclassification
    class_weights = torch.tensor([1.0, 4.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    model = OracleLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("[*] Initializing Oracle LSTM Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch['meta'].to(device)), batch['targets'].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                _, preds = torch.max(model(batch['meta'].to(device)), 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch['targets'].cpu().numpy())
                
        acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
        print(f"    -> Epoch {epoch+1:02d} | Loss: {total_loss/max(1, len(train_loader)):.4f} | Val Acc: {acc:.4f}")

    print("[*] Training successfully completed. Finalizing validation.")

if __name__ == "__main__":
    train_oracle()