#!/usr/bin/env python3
"""
Debug script to test API authentication
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_auth():
    base_url = "http://localhost:8000"
    api_keys = os.getenv("API_KEYS", "")
    
    print(f"API_KEYS from .env: {api_keys}")
    
    if not api_keys:
        print("❌ No API_KEYS found in environment")
        return
    
    # Parse the first API key
    first_key = api_keys.split(',')[0].split(':')[0].strip()
    print(f"Using API key: {first_key}")
    
    # Test with health endpoint first (should work without auth)
    print("\n1. Testing health endpoint (no auth required)...")
    try:
        response = requests.get(f"{base_url}/api/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Health endpoint works")
        else:
            print(f"❌ Health failed: {response.text}")
    except Exception as e:
        print(f"❌ Health failed: {e}")
    
    # Test with chunk endpoint (requires auth)
    print(f"\n2. Testing chunk endpoint with API key: {first_key[:8]}...")
    try:
        headers = {"Authorization": f"Bearer {first_key}"}
        data = {
            "text": "This is a test text for authentication.",
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": 95
        }
        
        response = requests.post(f"{base_url}/api/chunk", json=data, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Authentication successful!")
        elif response.status_code == 401:
            print("❌ Authentication failed")
        else:
            print(f"⚠️ Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_auth()