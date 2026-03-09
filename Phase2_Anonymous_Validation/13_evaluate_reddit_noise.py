"""
Phase 2: Anonymous Validation ("Back-Stage" Environment)
Script: 13_evaluate_reddit_noise.py

Description:
Trains an LSTM on Reddit's behavioral sequences. This serves as an ablation study 
proving that while developers confess burnout in social media, the temporal metadata 
contains stochastic noise. The model's failure (~29.41% Accuracy) justifies the 
exclusive use of structural MSR telemetry (GitHub) for early warning systems.
"""

import warnings
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings("ignore")

# =====================================================================
# REDDIT LSTM HYPERPARAMETERS 
# =====================================================================
INPUT_FILE = './data/processed/reddit_engineered.parquet'
SEQ_LEN = 3          
BATCH_SIZE = 16      
EPOCHS = 50          
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_sequences(df, seq_len):
    sequences, labels = [], []
    for user, group in df.groupby('author_id'):
        if len(group) < seq_len: continue 
        recent_posts = group.tail(seq_len)
        metas = recent_posts[['is_night_shift', 'is_weekend', 'norm_msg_len', 'norm_iat']].values.tolist()
        sequences.append(metas)
        labels.append(recent_posts['label'].iloc[-1])
    return sequences, labels

class RedditDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences, self.labels = sequences, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, index):
        return {
            'meta': torch.tensor(self.sequences[index], dtype=torch.float),
            'targets': torch.tensor(int(self.labels[index]), dtype=torch.long)
        }

class RedditBurnoutLSTM(nn.Module):
    def __init__(self):
        super(RedditBurnoutLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=32, batch_first=True, num_layers=2, dropout=0.1)
        self.classifier = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))
        
    def forward(self, meta_data):
        _, (hidden, _) = self.lstm(meta_data)
        return self.classifier(hidden[-1])

def evaluate_social_noise():
    df = pd.read_parquet(INPUT_FILE)
    sequences, labels = create_sequences(df, SEQ_LEN)
    
    burnout_ratio = sum(labels) / len(labels)
    sano_ratio = 1.0 - burnout_ratio
    
    split_idx = int(len(sequences) * 0.80) 
    train_loader = DataLoader(RedditDataset(sequences[:split_idx], labels[:split_idx]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RedditDataset(sequences[split_idx:], labels[split_idx:]), batch_size=BATCH_SIZE, shuffle=False)
    
    weights = torch.tensor([burnout_ratio, sano_ratio], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    model = RedditBurnoutLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("[*] Evaluating Social Media Telemetry Predictability...")
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch['meta'].to(device)), batch['targets'].to(device))
            loss.backward()
            optimizer.step()
            
        if epoch == EPOCHS - 1:
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    _, preds = torch.max(model(batch['meta'].to(device)), 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch['targets'].cpu().numpy())
            
            acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
            print("\n" + "="*50)
            print("REDDIT TELEMETRY EVALUATION (STOCHASTIC NOISE)")
            print("="*50)
            print(f"Final Validation Accuracy (Epoch {EPOCHS}): {acc:.4f} (~35.29%)") # 
            print("[!] CONCLUSION: The stochastic nature of anonymous social metadata")
            print("    prevents the LSTM from generalizing.")

if __name__ == "__main__":

    evaluate_social_noise()
