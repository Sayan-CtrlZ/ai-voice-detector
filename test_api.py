import requests
import base64

# 1. Pick a file to test (Change this filename to test different ones!)
filename = "dataset/real/real1.wav" 

# 2. Convert to Base64
with open(filename, "rb") as f:
    audio_data = f.read()
    b64_string = base64.b64encode(audio_data).decode("utf-8")

# 3. Send to your API
url = "http://127.0.0.1:8000/detect"
payload = {"audio_base64": b64_string, "language": "en"}

print(f"Testing with {filename}...")
response = requests.post(url, json=payload)
print("\nRESULT:", response.json())