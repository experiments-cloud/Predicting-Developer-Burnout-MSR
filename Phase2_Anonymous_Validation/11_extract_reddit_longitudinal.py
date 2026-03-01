"""
Phase 2: Anonymous Validation ("Back-Stage" Environment)
Script: 11_extract_reddit_longitudinal.py

Description:
Extracts the temporal digital footprint of users identified in the semantic phase.
By retrieving their historical comments across the platform, it builds the 
longitudinal sequences required to test behavioral telemetry in social networks.
"""

import os
import praw
import pandas as pd
from datetime import datetime

# =====================================================================
# LONGITUDINAL EXTRACTION HYPERPARAMETERS 
# =====================================================================
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent="MSR_Burnout_Research_v1.0"
)

USERS_PER_CLASS = 50       
HISTORY_LIMIT = 15         
OUTPUT_FILE = './data/raw/reddit_longitudinal_raw.parquet'

def get_user_history(username, label):
    history = []
    try:
        user = reddit.redditor(username)
        for comment in user.comments.new(limit=HISTORY_LIMIT):
            history.append({
                'author_id': username,
                'date': datetime.utcfromtimestamp(comment.created_utc),
                'text_length': len(comment.body.split()),
                'subreddit': comment.subreddit.display_name,
                'label': label 
            })
    except Exception:
        pass # Handle deleted/private accounts silently
    return history

def extract_reddit_timelines():
    print("[*] Initializing Longitudinal Social Extractor...")
    all_data = []
    
    # Placeholder logic mirroring the manuscript's heuristic targeting
    print("[*] Targeting developers with explicit Burnout indicators...")
    burnout_users = set()
    for sub in reddit.subreddit("cscareerquestions+ITCareerQuestions").search("burnout OR exhausted", limit=200):
        if sub.author and sub.author.name not in burnout_users:
            burnout_users.add(sub.author.name)
            if len(burnout_users) >= USERS_PER_CLASS: break

    for user in burnout_users:
        all_data.extend(get_user_history(user, label=1))

    print("[*] Targeting stable Control developers...")
    control_users = set()
    for sub in reddit.subreddit("Python+learnprogramming").search("tutorial OR success OR project", limit=200):
        if sub.author and sub.author.name not in burnout_users and sub.author.name not in control_users:
            control_users.add(sub.author.name)
            if len(control_users) >= USERS_PER_CLASS: break

    for user in control_users:
        all_data.extend(get_user_history(user, label=0))

    if all_data:
        df = pd.DataFrame(all_data).sort_values(['author_id', 'date'])
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_parquet(OUTPUT_FILE, index=False)
        print(f"[*] Extraction complete. Saved {len(df)} historical interactions.")

if __name__ == "__main__":
    extract_reddit_timelines()