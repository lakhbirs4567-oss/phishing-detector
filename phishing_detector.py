import os
import re
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# 1. Environment Configuration
load_dotenv()
GITHUB_TOKEN = os.getenv("GH_PAT") or os.getenv("GITHUB_TOKEN")


# 2. Machine Learning Feature Extraction
def extract_features(text):
    """Extract metadata features from email text."""
    url_count = len(re.findall(r'https?://\S+|www\.\S+', text))
    has_ip = 1 if re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text) else 0
    
    keywords = ['urgent', 'verify', 'account', 'suspended', 'login', 'bank', 'password', 'update', 'click', 'expire', 'security']
    keyword_count = sum(text.lower().count(kw) for kw in keywords)
    email_length = len(text)
    
    return [url_count, has_ip, keyword_count, email_length]


# 3. GitHub API Integration
def check_github_repo(owner, repo):
    """Query GitHub API using the stored personal access token."""
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}" if GITHUB_TOKEN else ""
    }
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print(f"[GitHub API] Repo Name: {data.get('full_name')}")
        print(f"[GitHub API] Stars: {data.get('stargazers_count')}")
        return data
    else:
        print(f"[GitHub API Error] {response.status_code}: {response.json().get('message')}")
        return None


# 4. Main Execution
if __name__ == "__main__":
    print("--- 1. Testing GitHub API Integration ---")
    check_github_repo("lakhbirs4567-oss", "phishing-detector")

    print("\n--- 2. Running Phishing Detection Model ---")
    sample_email = "URGENT: Your bank account has been suspended! Click http://192.168.1.1/login to update password."
    features = extract_features(sample_email)
    
    print(f"Sample Input: {sample_email[:50]}...")
    print(f"Extracted Features (URLs, IP, Keywords, Length): {features}")
