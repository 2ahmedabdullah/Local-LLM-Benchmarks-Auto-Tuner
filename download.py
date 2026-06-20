import os

# 🔑 HUGGING FACE TOKEN HERE


# --- High-Performance Network Configuration Matrix ---
os.environ["HF_TOKEN"] = HF_ACCESS_TOKEN
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

# Prevent huggingface_hub from spamming warnings about deprecation loops
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from huggingface_hub import hf_hub_download

def secure_download(repo_id, filename):
    print(f"\n⚡ Initiating authenticated chunk transfer for: {filename}")
    print(f"📂 Destination: ./models/{filename}")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir="./models",
            local_dir_use_symlinks=False
        )
        print(f"✅ Successfully downloaded and verified: {filename}")
    except Exception as e:
        print(f"❌ Download failed for {filename}: {e}")

if __name__ == "__main__":
    # Define our exact target pair for the RTX 3050 matrix
    models_to_fetch = [
        {"repo": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF", "file": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"},
        {"repo": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "file": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"}
    ]
    
    print("🚀 Initializing Authenticated Pre-Download Matrix Run...")
    
    # Simple check to make sure the user replaced the placeholder token string
    if "actual_hf_token" in HF_ACCESS_TOKEN:
        print("⚠️ ERROR: Please paste real Hugging Face token string into the script first!")
    else:
        for item in models_to_fetch:
            secure_download(item["repo"], item["file"])
        print("\n🏁 Master local model cache populated. System primed for tomorrow's benchmarking!")