"""
Phase 3: Behavioral Telemetry in Version Control (GitHub)
Script: 06_extract_github_mass.py

Description:
Executes the massive extraction of version control telemetry using the GitHub REST API.
It targets high-pressure environments (both corporate and open-source) to build a robust 
dataset of 119,486 commits. It explicitly discards semantic data (commit messages) to 
prevent social desirability bias, focusing purely on operational friction (Section 3.5.1).
"""

import os
import time
import requests
import pandas as pd

# =====================================================================
# MASS EXTRACTION HYPERPARAMETERS (Section 3.5.1)
# =====================================================================
# GitHub Personal Access Token must be set as an environment variable for security
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

# Target high-cognitive-load repositories ensuring ecological validity
REPOSITORIES = [
    'facebook/react',
    'tensorflow/tensorflow',
    'microsoft/vscode',
    'django/django',       
    'torvalds/linux',      
    'bitcoin/bitcoin'      
]

MAX_COMMITS_PER_REPO = 20000 
OUTPUT_FILE = './data/raw/github_telemetry_mass.parquet'

HEADERS = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def check_rate_limit(response):
    """Handles GitHub API rate limits dynamically to ensure uninterrupted massive extraction."""
    if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
        if int(response.headers['X-RateLimit-Remaining']) == 0:
            reset_time = int(response.headers['X-RateLimit-Reset'])
            sleep_time = max(0, reset_time - int(time.time())) + 10 
            print(f"\n[!] API Limit Reached. Sleeping for {sleep_time / 60:.1f} minutes...")
            time.sleep(sleep_time)
            print("[*] Waking up and resuming extraction...")
            return True
    return False

def extract_mass_commits():
    if not GITHUB_TOKEN:
        print("[!] ERROR: GITHUB_TOKEN environment variable is not set.")
        return

    print("[*] Initializing Massive MSR Telemetry Extractor...")
    os.makedirs('./data/raw', exist_ok=True)
    all_data = []

    for repo in REPOSITORIES:
        print(f"\n[*] Mining Repository: {repo}")
        commits_collected = 0
        page = 1

        while commits_collected < MAX_COMMITS_PER_REPO:
            url_list = f'https://api.github.com/repos/{repo}/commits?per_page=100&page={page}'
            res_list = requests.get(url_list, headers=HEADERS)
            
            if check_rate_limit(res_list): continue
            if res_list.status_code != 200: break
                
            commits = res_list.json()
            if not commits: break

            for item in commits:
                if commits_collected >= MAX_COMMITS_PER_REPO: break
                
                # Skip authorless commits to optimize API quota usage
                if not item.get('author'): continue
                
                sha = item['sha']
                url_detail = f'https://api.github.com/repos/{repo}/commits/{sha}'
                res_detail = requests.get(url_detail, headers=HEADERS)
                
                if check_rate_limit(res_detail):
                    res_detail = requests.get(url_detail, headers=HEADERS)

                if res_detail.status_code == 200:
                    detail = res_detail.json()
                    author_login = detail['author']['login'] if detail.get('author') else 'unknown'
                    
                    # Note: We extract text here temporarily, but it is dropped in the next 
                    # pipeline stage to guarantee pure behavioral modeling.
                    all_data.append({
                        'repo': repo,
                        'author_id': author_login,
                        'date': item['commit']['author']['date'],
                        'commit_message': item['commit']['message'],
                        'lines_added': detail.get('stats', {}).get('additions', 0),
                        'lines_deleted': detail.get('stats', {}).get('deletions', 0)
                    })
                    
                    commits_collected += 1
                    if commits_collected % 500 == 0:
                        print(f"    -> {commits_collected}/{MAX_COMMITS_PER_REPO} commits extracted.")
            page += 1

    print(f"\n[*] Extraction complete. {len(all_data)} total records retrieved.")
    if all_data:
        pd.DataFrame(all_data).to_parquet(OUTPUT_FILE, index=False)
        print(f"[*] Raw telemetry saved safely to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_mass_commits()