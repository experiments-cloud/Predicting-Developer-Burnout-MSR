"""
Phase 4: Macroeconomic Context & Ecological Validity
Script: 14_extract_hackernews_mass.py

Description:
Employs a "Time-Travel" pagination heuristic via timestamps to bypass the standard 
Algolia API limits. This extracts a massive historical corpus (20,000 records) of 
burnout discussions from Hacker News. It contextualizes the behavioral findings 
within broader macroeconomic industry trends.
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime

# =====================================================================
# MASS EXTRACTION CONFIGURATION
# =====================================================================
QUERY = "burnout"
TARGET_RECORDS = 20000
HITS_PER_PAGE = 100
OUTPUT_FILE = './data/raw/hackernews_mass_raw.parquet'

def extract_hn_mass():
    print(f"[*] Initializing Time-Travel Mass Extraction for '{QUERY}'...")
    os.makedirs('./data/raw', exist_ok=True)
    
    all_comments = []
    last_timestamp = None
    
    while len(all_comments) < TARGET_RECORDS:
        url = f"http://hn.algolia.com/api/v1/search_by_date?query={QUERY}&tags=comment&hitsPerPage={HITS_PER_PAGE}"
        
        if last_timestamp:
            url += f"&numericFilters=created_at_i<{last_timestamp}"
            
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"[!] API Limit Reached: HTTP {response.status_code}. Sleeping 10s...")
                time.sleep(10)
                continue
                
            data = response.json()
            hits = data.get('hits', [])
            
            if not hits:
                print("[*] Historical boundary reached. Terminating extraction.")
                break 
                
            for item in hits:
                if not any(c['id'] == item.get('objectID') for c in all_comments):
                    all_comments.append({
                        'id': item.get('objectID'),
                        'author_id': item.get('author'),
                        'created_at': item.get('created_at'),
                        'text': item.get('comment_text', ''),
                        'story_id': item.get('story_id')
                    })
            
            last_timestamp = hits[-1]['created_at_i']
            oldest_date = datetime.utcfromtimestamp(last_timestamp).strftime('%Y-%m-%d')
            print(f"    -> {len(all_comments)}/{TARGET_RECORDS} extracted. Traveling back to: {oldest_date}")
            
            time.sleep(0.8) # Polite API pacing
            
        except requests.exceptions.RequestException as e:
            print(f"[!] Network Error: {e}. Retrying...")
            time.sleep(5)

    if all_comments:
        df = pd.DataFrame(all_comments)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.drop_duplicates(subset=['id']).sort_values(['author_id', 'created_at'])
        
        df.to_parquet(OUTPUT_FILE, index=False)
        print(f"[*] Success! {len(df)} historical records saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_hn_mass()