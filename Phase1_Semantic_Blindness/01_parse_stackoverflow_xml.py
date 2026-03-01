"""
Phase 1: Semantic Blindness in Formal Environments (Stack Overflow)
Script: 01_parse_stackoverflow_xml.py

Description:
Parses the massive Stack Overflow XML dump efficiently using iterative parsing.
Crucially, it implements the cryptographic anonymization protocol (SHA-256 with 
dynamic salt) to protect developer identities, adhering to strict research ethics 
(Section 3.3). It also performs initial HTML sanitization while preserving the 
structural [CODE_BLOCK] tokens.
"""

import os
import re
import csv
import secrets
import hashlib
import gc
import xml.etree.ElementTree as etree
from datetime import datetime

# =====================================================================
# EXTRACTION & SECURITY CONFIGURATION (Section 3.3)
# =====================================================================
INPUT_FILE = './data/raw/post.xml'
OUTPUT_FILE = './data/processed/dataset_anonymized.csv'
SALT_FILE = './data/security/salt.key'
MIN_YEAR = 2018

# Pre-compiled Regex for faster execution on large datasets
RE_CODE_BLOCK = re.compile(r'<code>(.*?)</code>', re.DOTALL)
RE_HTML_TAGS = re.compile(r'<[^>]+>')
RE_WHITESPACE = re.compile(r'\s+')
RE_URL = re.compile(r'http\S+')

def load_dynamic_salt():
    """Loads or generates a dynamic salt for cryptographic hashing."""
    os.makedirs(os.path.dirname(SALT_FILE), exist_ok=True)
    if os.path.exists(SALT_FILE):
        with open(SALT_FILE, 'rb') as f:
            return f.read()
    salt = secrets.token_bytes(32)
    with open(SALT_FILE, 'wb') as f:
        f.write(salt)
    return salt

def hash_user_id(user_id, salt):
    """Applies SHA-256 hashing to guarantee developer anonymity."""
    if not user_id: return "unknown"
    payload = str(user_id).encode('utf-8') + salt
    return hashlib.sha256(payload).hexdigest()[:16]

def calculate_code_ratio(text):
    if not text: return 0.0
    code_matches = RE_CODE_BLOCK.findall(text)
    code_len = sum(len(c) for c in code_matches)
    total_len = len(text)
    return round(code_len / total_len, 4) if total_len > 0 else 0.0

def sanitize_html(text):
    if not text: return ""
    text = RE_CODE_BLOCK.sub(' [CODE_BLOCK] ', text)
    text = RE_HTML_TAGS.sub('', text)
    text = RE_URL.sub('[URL]', text)
    return RE_WHITESPACE.sub(' ', text).strip()

def process_massive_xml():
    salt = load_dynamic_salt()
    print(f"[*] Initializing iterative XML parsing from: {INPUT_FILE}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_hash', 'date', 'hour', 'score', 'code_ratio', 'text'])

        try:
            context = etree.iterparse(INPUT_FILE, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)

            count = 0
            for event, elem in context:
                if event == 'end' and elem.tag == 'row':
                    uid = elem.get('OwnerUserId')
                    date_str = elem.get('CreationDate')
                    body = elem.get('Body', '')
                    score = elem.get('Score', 0)

                    if uid and date_str:
                        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
                        if dt.year >= MIN_YEAR:
                            writer.writerow([
                                hash_user_id(uid, salt),
                                date_str,
                                dt.hour,
                                score,
                                calculate_code_ratio(body),
                                sanitize_html(body)
                            ])
                            count += 1
                            if count % 100000 == 0:
                                print(f"    -> Processed: {count} records...")
                    
                    # Clear memory to prevent RAM exhaustion on massive XMLs
                    root.clear()
                
            del context, root
            gc.collect()
            print(f"[*] Success! Anonymized dataset saved. Total records: {count}")

        except Exception as e:
            print(f"[!] CRITICAL ERROR: {e}")

if __name__ == "__main__":
    process_massive_xml()