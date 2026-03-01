"""
Phase 1: Semantic Blindness in Formal Environments (Stack Overflow)
Script: 05_evaluate_multimodal_so.py

Description:
Evaluates the DistilBERT Multimodal architecture on the hold-out test set.
It generates the classification report and the confusion matrix, empirically 
demonstrating the "Semantic Blindness" phenomenon (Section 4.2). The model 
converges to an accuracy equivalent to a random classifier (~48%), validating 
the presence of severe Professional Masking in formal technical forums.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================================
# EVALUATION CONFIGURATION (Section 4.2)
# =====================================================================
TEST_FILE = './data/processed/test_data.parquet'
MODEL_PATH = './models/burnout_model.pth'
OUTPUT_PLOT = './results/confusion_matrix_so.png'

MAX_LEN = 128
BATCH_SIZE = 32
EVALUATION_SAMPLE = 100000 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StackOverflowDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self): return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = str(row['text'])
        inputs = self.tokenizer.encode_plus(
            text, None, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', return_token_type_ids=False,
            truncation=True, return_attention_mask=True, return_tensors='pt'
        )
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

class BurnoutClassifier(nn.Module):
    """Reconstructs the Late-Fusion multimodal architecture for inference."""
    def __init__(self):
        super(BurnoutClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_layer = nn.Linear(768, 64)
        self.meta_layer = nn.Linear(3, 16)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(80, 2))
        
    def forward(self, input_ids, attention_mask, meta_data):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = bert_output.last_hidden_state[:, 0]
        text_vec = torch.relu(self.text_layer(hidden_state))
        meta_vec = torch.relu(self.meta_layer(meta_data))
        return self.classifier(torch.cat((text_vec, meta_vec), dim=1))

def evaluate_model():
    print(f"[*] Loading hold-out test set (Sample: {EVALUATION_SAMPLE})...")
    df_test = pd.read_parquet(TEST_FILE)
    if len(df_test) > EVALUATION_SAMPLE:
        df_test = df_test.sample(EVALUATION_SAMPLE, random_state=42)
        
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    test_loader = DataLoader(StackOverflowDataset(df_test, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
    
    model = BurnoutClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    all_targets, all_preds = [], []
    
    print("[*] Running inference on formal environment data...")
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['ids'].to(device), batch['mask'].to(device), batch['meta'].to(device))
            _, preds = torch.max(outputs, dim=1)
            all_targets.extend(batch['targets'].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT (Semantic Blindness Evidence)")
    print("="*50)
    print(classification_report(all_targets, all_preds, target_names=['Active (0)', 'Churn/Burnout (1)']))
    
    print("[*] Rendering Semantic Blindness Confusion Matrix...")
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(all_targets, all_preds), annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred. Active', 'Pred. Churn'], yticklabels=['True Active', 'True Churn'])
    plt.title('Multimodal Semantic Model - Confusion Matrix')
    plt.ylabel('True Clinical State (Ground Truth)')
    plt.xlabel('Algorithm Prediction')
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"[*] Visual artifact saved at: {OUTPUT_PLOT}")

if __name__ == "__main__":
    evaluate_model()