#!/usr/bin/env python3
"""
Test script to verify the API endpoints work
"""

import requests
import json
import time

def test_api():
    base_url = "http://localhost:8001"
    
    print("Testing Semantic Chunking Service API...")
    
    # Test 1: Health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint works!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint works!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test 3: Chunk endpoint (this will fail without real API key, but we can test validation)
    print("\n3. Testing chunk endpoint validation...")
    try:
        test_data = {
            "text": "This is a test text for semantic chunking.",
            "breakpoint_threshold_type": "percentile",
            "breakpoint_threshold_amount": 95
        }
        
        response = requests.post(
            f"{base_url}/api/chunk", 
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Chunk endpoint works!")
            result = response.json()
            print(f"Created {len(result['chunks'])} chunks")
        elif response.status_code == 500:
            print("⚠️  Chunk endpoint reached processing (expected without OpenAI API key)")
            print(f"Error: {response.json()}")
        else:
            print(f"❌ Chunk endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Chunk endpoint failed: {e}")

if __name__ == "__main__":
    # Wait a moment for the server to be ready
    time.sleep(2)
    test_api()