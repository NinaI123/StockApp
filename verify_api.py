import sys
import os
import json
import logging

# Ensure project root is in path
sys.path.append(os.getcwd())

from app import app

def test_api():
    client = app.test_client()
    
    print("\n--- Testing /api/stock/analysis ---")
    response = client.get('/api/stock/analysis?symbol=AAPL')
    if response.status_code == 200:
        data = response.get_json()
        print("Success!")
        print(json.dumps(data, indent=2))
    else:
        print(f"Failed: {response.status_code}")
        print(response.get_data(as_text=True))

    print("\n--- Testing /api/model/performance ---")
    response = client.get('/api/model/performance')
    if response.status_code == 200:
        data = response.get_json()
        print("Success!")
        print(json.dumps(data, indent=2))
    else:
        print(f"Failed: {response.status_code}")
        
    print("\n--- Testing /api/backtest ---")
    response = client.post('/api/backtest', json={
        "symbol": "AAPL",
        "strategy": "momentum"
    })
    if response.status_code == 200:
        data = response.get_json()
        print("Success!")
        print(json.dumps(data, indent=2))
    else:
        print(f"Failed: {response.status_code}")

    print("\n--- Testing /api/insights/daily ---")
    response = client.get('/api/insights/daily')
    if response.status_code == 200:
        data = response.get_json()
        print("Success!")
        print(json.dumps(data, indent=2))
    else:
        print(f"Failed: {response.status_code}")
        # 503 is acceptable if SPY fetch fails, but structure should be correct

if __name__ == "__main__":
    test_api()
