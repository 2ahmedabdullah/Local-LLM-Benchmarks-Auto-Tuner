import os

# 🔑 PASTE the HUGGING FACE TOKEN HERE
HF_ACCESS_TOKEN = "ABC"


# --- High-Performance Network Configuration Matrix ---
os.environ["HF_TOKEN"] = HF_ACCESS_TOKEN
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

# Prevent huggingface_hub from spamming warnings about deprecation loops
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from huggingface_hub import hf_hub_download, snapshot_download

def secure_download(repo_id, filename=None):
    if filename:
        # Download a single specific file (e.g., GGUF)
        print(f"\n⚡ Initiating authenticated chunk transfer for file: {filename}")
        print(f"📂 Destination: ./models/{filename}")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir="./models",
            )
            print(f"✅ Successfully downloaded and verified file: {filename}")
        except Exception as e:
            print(f"❌ Download failed for {filename}: {e}")
    else:
        # Download the entire full-precision repository (Safetensors shards + configs)
        # We append the repo name to the path so files don't get messy
        repo_folder_name = repo_id.split("/")[-1]
        destination = f"./models/{repo_folder_name}"
        
        print(f"\n⚡ Initiating full repository snapshot transfer for: {repo_id}")
        print(f"📂 Destination: {destination}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=destination,
            )
            print(f"✅ Successfully downloaded and verified full repository: {repo_id}")
        except Exception as e:
            print(f"❌ Full repository download failed for {repo_id}: {e}")

if __name__ == "__main__":
    # Define our exact target pair for the RTX 3050 matrix
    models_to_fetch = [
    # --- UNQUANTIZED / BASE INSTRUCT MODEL ---
    # Ideal for native Hugging Face / Transformers, Unsloth training, or 16-bit inference
    # {
    #     "repo": "unsloth/Qwen2.5-VL-3B-Instruct", 
    #     "file": None  # Load full repository weights (Safetensors)
    # },
    
    # High Fidelity Q8 (Near unquantized performance, larger file size)
    {
        "repo": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF", 
        "file": "Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
    },
    ]
    
    print("🚀 Initializing Authenticated Pre-Download Matrix Run...")
    
    # Simple check to make sure the user replaced the placeholder token string
    if "the_actual_hf_token" in HF_ACCESS_TOKEN:
        print("⚠️ ERROR: Please paste the real Hugging Face token string into the script first!")
    else:
        for item in models_to_fetch:
            secure_download(item["repo"], item["file"])
        print("\n🏁 Master local model cache populated. System primed for tomorrow's benchmarking!")