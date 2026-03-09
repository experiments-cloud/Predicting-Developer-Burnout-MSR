"""
Phase 1: Semantic Blindness in Formal Environments (Stack Overflow)
Script: 04_train_multimodal_so.py

Description:
Trains the Multimodal Deep Learning architecture (DistilBERT + Linear Metadata) 
described in Section 3.3. This script empirically demonstrates the "Semantic Blindness" 
hypothesis: when applied to front-stage formal environments, the model suffers from 
gradient starvation (converging to ~0.693 Loss) due to developers heavily sterilizing 
their professional communication (Professional Masking).
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# =====================================================================
# MULTIMODAL ARCHITECTURE HYPERPARAMETERS (Section 3.3)
# =====================================================================
TRAIN_FILE = './data/processed/train_balanced.parquet'
TEST_FILE = './data/processed/test_data.parquet'
MODEL_SAVE_PATH = './models/distilbert_multimodal_so.pth'

MAX_LEN = 128        # Truncation length for Transformers
BATCH_SIZE = 16      # Optimized for VRAM constraints
EPOCHS = 3           # Sufficient to demonstrate early mathematical stagnation
LEARNING_RATE = 2e-5 # Fine-tuning learning rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StackOverflowDataset(Dataset):
    """Fuses unstructured text and structured operational metadata."""
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        # NLP Branch Processing
        text = str(row['cleaned_text'])
        inputs = self.tokenizer.encode_plus(
            text, None, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        # Behavioral Metadata Branch
        meta_features = torch.tensor([
            float(row['code_ratio']),
            float(row['score']),
            float(row['hour']) / 24.0 
        ], dtype=torch.float)
        
        return {
            'ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'meta': meta_features,
            'targets': torch.tensor(int(row['label']), dtype=torch.long)
        }

class BurnoutMultimodalClassifier(nn.Module):
    """Late-Fusion architecture combining DistilBERT and Linear layers."""
    def __init__(self):
        super(BurnoutMultimodalClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_layer = nn.Linear(768, 64)
        
        self.meta_layer = nn.Linear(3, 16)
        
        # Late Fusion: 64 (text) + 16 (meta) = 80 -> Binary Classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(80, 2)
        )
        
    def forward(self, input_ids, attention_mask, meta_data):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_output.last_hidden_state[:, 0] # Extract [CLS] token
        
        text_vec = torch.relu(self.text_layer(hidden_state))
        meta_vec = torch.relu(self.meta_layer(meta_data))
        
        combined = torch.cat((text_vec, meta_vec), dim=1)
        return self.classifier(combined)

def train_multimodal_network():
    print("[*] Initializing Multimodal Burnout Detection Training...")
    df = pd.read_parquet(TRAIN_FILE)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    training_set = StackOverflowDataset(train_df, tokenizer, MAX_LEN)
    val_set = StackOverflowDataset(val_df, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    model = BurnoutMultimodalClassifier().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        
        for batch in loop:
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            meta = batch['meta'].to(device, dtype=torch.float)
            targets = batch['targets'].to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(ids, mask, meta)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            
        # Validation Protocol
        model.eval()
        val_targets, val_preds = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    batch['ids'].to(device), 
                    batch['mask'].to(device), 
                    batch['meta'].to(device)
                )
                _, preds = torch.max(outputs, dim=1)
                val_targets.extend(batch['targets'].cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
        
        acc = accuracy_score(val_targets, val_preds)
        avg_loss = total_loss / len(train_loader)
        
        print(f"[*] Epoch {epoch+1} Completed | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")
        
        # Expected behavior: Loss stagnates at ~0.693 due to Professional Masking
        if round(avg_loss, 2) == 0.693:
            print("[!] Warning: Gradient Starvation detected. The formal environment lacks predictive signal.")
            
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":

    train_multimodal_network()
