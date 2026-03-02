import requests
import json

print("="*60)
print("🔍 TESTING HISTORY PAGE ENDPOINTS")
print("="*60)

base_url = "http://127.0.0.1:5000"

endpoints = [
    "/api/performance_metrics",
    "/api/class_metrics", 
    "/history/data",
    "/api/tts-status"
]

for endpoint in endpoints:
    try:
        print(f"\n📡 Testing {endpoint}...")
        response = requests.get(base_url + endpoint, timeout=5)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"   ✅ Success! Response type: {type(data)}")
                if isinstance(data, list):
                    print(f"   📊 Data length: {len(data)}")
                    if len(data) > 0:
                        print(f"   First item: {json.dumps(data[0], indent=2)}")
                elif isinstance(data, dict):
                    print(f"   📊 Keys: {list(data.keys())}")
                    for key in list(data.keys())[:3]:
                        if data[key]:
                            if isinstance(data[key], list):
                                print(f"      {key}: {len(data[key])} items")
                            else:
                                print(f"      {key}: {data[key]}")
            except:
                print(f"   ❌ Response is not JSON: {response.text[:100]}")
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout - server took too long to respond")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*60)