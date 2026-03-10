# test_models.py
from app.models import check_available_models, get_device

print("🔍 Checking available models...")
available = check_available_models()

for model_name, is_available in available.items():
    status = "✅" if is_available else "❌"
    print(f"{status} {model_name}")

print(f"\n📡 Device: {get_device()}")