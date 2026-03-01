"""
Phase 1: Semantic Blindness in Formal Environments (Stack Overflow)
Script: 01_feature_extraction_so.py

Description:
Performs feature engineering on the raw Stack Overflow dataset.
It extracts structural metrics such as the Code Ratio and normalizes the 
Inter-Arrival Time (IAT). It also sanitizes the HTML content while preserving 
the structural markers for the DistilBERT multimodal architecture (Section 3.3).
"""

import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# =====================================================================
# I/O CONFIGURATION 
# =====================================================================
INPUT_FILE = './data/raw/stackoverflow_raw.csv'
OUTPUT_FILE = './data/processed/so_multimodal_features.csv'

class BehavioralFeatureEngineer:
    def __init__(self, df):
        self.df = df
        
    def _get_code_ratio(self, html_text):
        """
        Calculates the proportion of code blocks versus plain text.
        A high code ratio in formal environments often correlates with emotional 
        sterilization (Professional Masking).
        """
        if not isinstance(html_text, str): return 0.0
        
        code_blocks = re.findall(r'<code>(.*?)</code>', html_text, re.DOTALL)
        code_len = sum(len(c) for c in code_blocks)
        
        soup = BeautifulSoup(html_text, "html.parser")
        text_len = len(soup.get_text())
        
        if text_len == 0: return 0.0
        return code_len / (text_len + code_len)

    def _clean_text_keep_structure(self, html_text):
        """
        Sanitizes HTML but leaves a [CODE_BLOCK] token to allow the Transformer
        model to understand the structural context of the discussion.
        """
        if not isinstance(html_text, str): return ""
        
        soup = BeautifulSoup(html_text, "html.parser")
        
        for code in soup.find_all('code'):
            code.replace_with(' [CODE_BLOCK] ')
            
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process(self):
        print("[*] Parsing timestamps and sorting chronologically...")
        self.df['creation_date'] = pd.to_datetime(self.df['creation_date'])
        self.df = self.df.sort_values(['user_id', 'creation_date'])
        
        print("[*] Calculating normalized Inter-Arrival Time (IAT)...")
        self.df['iat_seconds'] = self.df.groupby('user_id')['creation_date'].diff().dt.total_seconds().fillna(0)
        self.df['iat_log'] = np.log1p(self.df['iat_seconds'])
        
        print("[*] Extracting structural metadata (Code Ratio)...")
        self.df['code_ratio'] = self.df['body'].apply(self._get_code_ratio)
        
        print("[*] Sterilizing text for DistilBERT processing...")
        self.df['cleaned_text'] = self.df['body'].apply(self._clean_text_keep_structure)
        
        final_cols = ['user_id', 'creation_date', 'cleaned_text', 'iat_log', 'code_ratio', 'label']
        return self.df[final_cols]

if __name__ == "__main__":
    print(f"[*] Loading raw data from {INPUT_FILE}...")
    try:
        df_raw = pd.read_csv(INPUT_FILE) 
        engineer = BehavioralFeatureEngineer(df_raw)
        df_processed = engineer.process()
        
        df_processed.to_csv(OUTPUT_FILE, index=False)
        print(f"[*] Success! Engineered dataset saved to: {OUTPUT_FILE}")
        
    except FileNotFoundError:
        print("[!] ERROR: Input file not found. Ensure the dataset is downloaded.")