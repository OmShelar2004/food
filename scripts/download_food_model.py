import os
import requests

MODEL_URL = "https://github.com/ansh941/MiniFood101/releases/download/v1.0/efficientnetb0_food101.keras"
MODEL_PATH = "tf_model_fixed.keras"

print("⬇️  Downloading EfficientNetB0 Food-101 model (~52 MB)...")

try:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(MODEL_URL, stream=True, allow_redirects=True, headers=headers, timeout=600)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Download complete! File size: {size_mb:.2f} MB")

    if size_mb < 40:
        print("⚠️ File seems incomplete — please re-run the script.")
    else:
        print("🎉 Model saved successfully at:", os.path.abspath(MODEL_PATH))

except Exception as e:
    print("❌ Download failed:", e)
