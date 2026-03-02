import requests
import time

print("="*60)
print("🔍 CHECKING IF FLASK SERVER IS RUNNING")
print("="*60)

url = "http://127.0.0.1:5000"

try:
    response = requests.get(url, timeout=3)
    print(f"✅ Server is running! Status code: {response.status_code}")
    print(f"✅ Response: {response.text[:100]}...")
except requests.exceptions.ConnectionError:
    print("❌ Server is NOT running!")
    print("\n📋 To start the server:")
    print("1. Open a new Command Prompt")
    print("2. cd C:\\Users\\HP PC\\Desktop\\ISL_Translator")
    print("3. venv\\Scripts\\activate")
    print("4. python web_app\\app.py")
    print("\nThen run this test again in a NEW window while server runs")
except Exception as e:
    print(f"❌ Error: {e}")