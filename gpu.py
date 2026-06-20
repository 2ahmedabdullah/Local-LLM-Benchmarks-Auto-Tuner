# gpu.py



import os
import ctypes
import sys
from fan_speed import cool_down

def setup_llama_cpp_cuda():
    # correct venv root
    base = sys.prefix

    llama_lib = os.path.join(base, "Lib", "site-packages", "llama_cpp", "lib")

    if not os.path.exists(llama_lib):
        raise RuntimeError(f"llama_cpp lib not found: {llama_lib}")

    # add DLL directories
    os.add_dll_directory(llama_lib)
    os.add_dll_directory(os.path.join(base, "Lib", "site-packages", "nvidia", "cublas", "bin"))
    os.add_dll_directory(os.path.join(base, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"))
    os.add_dll_directory(os.path.join(base, "Lib", "site-packages", "nvidia", "cuda_nvrtc", "bin"))
    os.add_dll_directory(os.path.join(base, "Lib", "site-packages", "nvidia", "cudnn", "bin"))

    # preload DLL chain
    for dll in [
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cpu.dll",
        "ggml-cuda.dll",
        "mtmd.dll",
        "llama.dll",
    ]:
        path = os.path.join(base, "Lib", "site-packages", "llama_cpp", "lib", dll)
        if os.path.exists(path):
            ctypes.CDLL(path)

setup_llama_cpp_cuda()

from llama_cpp import Llama
print("OK")


import re
import json
import pandas as pd
from datetime import datetime
import requests
import time
import csv

import pynvml 
import gc
import psutil
import warnings
from huggingface_hub import hf_hub_download
import uuid
import threading
import time
from datetime import datetime

# Initialize NVML once at the start of the script
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Device 0

warnings.filterwarnings("ignore")
warnings.getwarnings = lambda *args, **kwargs: None
from fan import get_instantaneous_fans, initialize_csv, continuous_thermal_logger

# --- CONFIGURATION PATHS ---
HWINFO_LOG_FILE = r"C:\Users\AbdulAhmed\Downloads\CPU vs GPU\hwdata.CSV"
CONFIG_FILE = "config.json"
TEST_IMAGE_PATH = r"cropped_visa_table.png"

os.environ["GGML_LOG_LEVEL"] = "ERROR"

# --- INITIALIZATION ---
try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_GPU_TRACKING = True
except Exception as e:
    print(f"⚠️ NVML Initialization Failed: {e}. Running without deep telemetry.")
    HAS_GPU_TRACKING = False

def get_true_vram():
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # Convert bytes directly to Gigabytes
    return info.used / (1024 ** 3)


def get_hwinfo_fans():
    """Scrapes the most recent physical fan RPM metrics recorded by HWiNFO."""
    if not os.path.exists(HWINFO_LOG_FILE):
        return "HWiNFO Log Not Found"
    
    try:
        with open(HWINFO_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return "Gathering data rows..."

        # Header row & Last row (most recent telemetry)
        headers = [h.strip().replace('"', '') for h in lines[0].split(",")]
        live_values = [v.strip().replace('"', '') for v in lines[-1].split(",")]

        fan_metrics = []
        for idx, header in enumerate(headers):
            if "fan" in header.lower() or "rpm" in header.lower():
                if idx < len(live_values) and live_values[idx]:
                    fan_metrics.append(f"{header}: {live_values[idx]} RPM")
                    
        return " | ".join(fan_metrics) if fan_metrics else "No active fan metrics found in headers."
    except Exception as e:
        return f"Error reading HWiNFO: {e}"
    
def get_peak_fans_in_window(generation_duration, log_interval_seconds=2):
    """
    Bypasses string-timestamp parsing entirely. Calculates how many log ticks 
    occurred during the model generation time and grabs the peak from those final rows.
    """
    if not os.path.exists(HWINFO_LOG_FILE):
        return "No data recorded (File missing)"
    
    try:
        with open(HWINFO_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return "Insufficient log rows to parse window."
            
        headers = [h.strip().replace('"', '') for h in lines[0].split(",")]
        
        # Calculate trailing rows to evaluate based on duration (plus a 2-row safety buffer)
        rows_to_scan = int(generation_duration / log_interval_seconds) + 2
        target_lines = lines[-rows_to_scan:]
        
        max_fan_speeds = {}
        
        for line in target_lines:
            vals = [v.strip().replace('"', '') for v in line.split(",")]
            if len(vals) < 2: 
                continue
                
            for idx, header in enumerate(headers):
                if "fan" in header.lower() or "rpm" in header.lower():
                    if idx < len(vals) and vals[idx]:
                        try:
                            rpm_val = float(vals[idx])
                            if header not in max_fan_speeds or rpm_val > max_fan_speeds[header]:
                                max_fan_speeds[header] = rpm_val
                        except ValueError:
                            pass # Skip non-numeric garbage data
                            
        if max_fan_speeds:
            return " | ".join([f"{k}: {int(v)} RPM" for k, v in max_fan_speeds.items()])
        return "No active hardware fan rows found in the trailing log buffer."
        
    except Exception as log_err:
        return f"Timeline row buffer scan anomaly: {log_err}"

     
def check_active_hardware():
    """Queries the local Ollama daemon to see what hardware is processing the model."""
    try:
        response = requests.get("http://localhost:11434/api/ps")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if not models:
                print("ℹ️ No models actively loaded in memory right now.")
                return
            for model in models:
                name = model.get("name")
                processor = model.get("processor", "Unknown")
                vram = model.get("size_vram", 0) / (1024**3) # Convert bytes to GB
                print(f"📊 Active Hardware Status -> Model: {name} | Backend Engine: {processor} | VRAM Used: {vram:.2f} GB")
        else:
            print("⚠️ Could not reach Ollama status endpoint.")
    except Exception as e:
        print(f"⚠️ Hardware check skipped: {e}")


def log_to_csv(metrics_dict, filename="benchmark_results.csv"):
    if not metrics_dict:
        return
    
    # Check if the file is currently open in Excel (PermissionError)
    try:
        # Check if file exists to handle header writing
        file_exists = os.path.isfile(filename)
        
        # 'a' (append) mode is safer than 'w'
        # 'newline=""' is CRITICAL to prevent empty rows on Windows
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
            
            # Only write header if the file is brand new
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(metrics_dict)
        print(f"✅ Telemetry committed to {filename}")
        
    except PermissionError:
        print(f"⚠️ CSV WARNING: {filename} is open in another program (e.g., Excel). Please close it.")
    except Exception as e:
        print(f"⚠️ CSV WARNING: Unexpected error: {e}")

# ==========================================
# EXTRACTION BLOCK SWAPPED TO LOCAL CHIP
# ==========================================
def fast_extract_table(params, RUN_ID):
            
    prompt = f"""
   "Provide an exhaustive, textbook-length historical breakdown of the Industrial Revolution "
"focusing on its progression between the years 1760 and 1840. the response must be an "
"extremely detailed, continuous academic narrative of at least 3000 words. Do not summarize."
"For EACH of the 10 points below, you must write multiple extensive paragraphs detailing "
"the exact technical mechanics, the immediate socio-economic disruptions, and the long-term historical legacy:\n\n"

    "the response must be a continuous, deeply detailed narrative that explicitly identifies, names, dates, and cross-references the following 10 historical milestones based *only* on their descriptions. 
    The items are listed completely out of chronological order:\n\n"

    "A. The landmark legislative act that first legally capped the daily work hours of miners/factory workers under the age of 13 to a maximum of 8 hours. (Identify the official name of the act and its exact year).\n"
    "B. The invention that automated the separation of seeds from fiber, which paradoxically caused a massive, measurable surge in the slave-labor economy of the American South. (Identify the inventor, the mechanism, and the patent year).\n"
    "C. The specific transition in metallurgy where an ironmaster replaced traditional timber-derived fuel with baked coal at a furnace in Shropshire, unlocking massive scalability. (Identify the ironmaster, the location, the fuel types, and the breakthrough year).\n"
    "D. The massive engineering marvel of 1830 that proved the commercial viability of inter-city passenger rail. You must name the specific locomotive that won the Rainhill Trials to service this line. (Identify the city-pair, the locomotive, and its designer).\n"
    "E. The demographic explosion of England's premier textile hub, explicitly detailing how its population scaled from roughly 10,000 in the early 1700s to its massive six-figure breakthrough by the early 1850s. (Name the city and the specific population figures).\n"
    "F. The addition of a separate cooling chamber to an existing atmospheric engine design, which drastically cut fuel consumption and allowed rotary motion. (Identify both the original designer, the modifier, the mechanism, and the modification patent year).\n"
    "G. The collective social uprising of textile artisans who smashed automated weaving machinery because it threatened their livelihoods. (Identify the specific term for these activists and the exact span of years the riots occurred).\n"
    "H. The grand international technology expo housed in a revolutionary iron-and-glass modular structure, serving as the definitive climax of this era. (Identify the name of the exhibition, the structure, and the exact year).\n"
    "I. The capital-investor and his specific Birmingham-based manufacturing plant that provided the financial backing, precision tooling, and commercial scaling required to deploy industrial steam utility. (Identify the partner and the manufactory name).\n"
    "J. The spinning machine driven by water power that shifted textile production entirely out of individual cottages and birthed the centralized factory system. (Identify the inventor, the machine name, and its patent year).\n\n" 
    
    "Ensure every single name, date, and statistic listed above is thoroughly integrated with deep historical context.

    "K. COMPREHENSIVE TEXTBOOK COMPLIANCE APPENDIX (MANDATORY BENCHMARK SOAK TASK)\n"
    "Immediately after completing the historical analysis above, you must generate a highly technical, "
    "word-for-word index glossary mapping every key mechanical term used in the response (e.g., atmospheric pressure, "
    "condensation, rolling cylinder, coke baking, rail draft friction, structural modules) to an extensive "
    "engineering definition. You must write out a minimum of 30 distinct glossary terms, detailing the absolute "
    "physics principles behind each. Do not shortcut this appendix. It must be a massive block of detailed technical prose."

        "Do not truncate or shorten the output."


    """
    
    try:
        
        # ⏱️ SYSTEM TIMESTAMP DECOUPLING
        _this_process = psutil.Process(os.getpid())

        inference_start_time = None       
        first_token_time = None           
        
        timeseries_log = []         
        window_start_time = None           
        tokens_in_current_second = 0
        current_second_bucket = 0.5

        # 1. Handle file downloading purely as a disk utility
        print("📥 Fetching/verifying model disk footprint...")
        model_local_path = hf_hub_download(
            repo_id=params.get("model_repo"),
            filename=params.get("model_filename"),
            local_dir="./models",
            local_dir_use_symlinks=False
        )

        # 2. FORCE A HARD PURGE OF PYTHON'S DISK BUFFER HEAP 🧹
        gc.collect()

        # 3. Reset the Host RAM baseline tracking right here
        _this_process = psutil.Process(os.getpid())
        base_ram_gb = _this_process.memory_info().rss / (1024 ** 3)

        print(f"💻 True Host RAM Baseline: {base_ram_gb:.2f} GB")

        print("🚀 Launching Direct-to-VRAM Hardware Pipeline...")
        # 4. Use the raw Llama constructor pointing directly to the file path
        llm = Llama(
            model_path=model_local_path,   
            n_gpu_layers=-1,               
            use_mmap=False,                 
            use_mlock=True,              
            split_mode=0,
            n_ctx=params.get("num_ctx", 8192),
            f16_kv=params.get("f16_kv", True),
            flash_attn=True,
            offload_kqv=True,
            verbose=False
        )
        # T2: Capture exact inference prefill start timestamp
        inference_start_time = time.time()
        window_start_time = inference_start_time  # Synchronize rolling telemetry window
        
        # 2. Run inference in stream mode
        stream_response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("num_predict", 4096),
            stream=True
            )
        
        full_text = ""
        total_tokens_generated = 0

        # 3. Process the stream (SINGLE UNIFIED LOOP - FIXED FOR TRUE TPS)
        for chunk in stream_response:
            chunk_time = time.time()
            
            # T3: Capture TTFT on the very first chunk
            if first_token_time is None:
                first_token_time = chunk_time
                
            # Collect content
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                chunk_content = delta['content']
                full_text += chunk_content
                
                # Tokenize the actual string chunk to get a true token count
                if chunk_content:
                    chunk_tokens = len(llm.tokenize(chunk_content.encode("utf-8")))
                else:
                    chunk_tokens = 0

                # Accumulate metrics
                tokens_in_current_second += chunk_tokens
                total_tokens_generated += chunk_tokens  # ◄── TRACKS OVERALL GENERATION TOTAL
                                
            # --- STOCHASTIC METRICS SAMPLING WINDOW (Every ~1 Second) ---
            elapsed_since_window = chunk_time - window_start_time
            if elapsed_since_window >= 0.5:
                rolling_tps = tokens_in_current_second / elapsed_since_window

                timeseries_log.append({
                    "second": current_second_bucket,
                    "instantaneous_tps": rolling_tps,
                })
                
                tokens_in_current_second = 0
                window_start_time = chunk_time
                current_second_bucket += 0.5
        
        print(f"Total Tokens Generated: {total_tokens_generated}")


        # ⏱️ CAPTURE TIMESTAMPS IMMEDIATELY AFTER THE STREAM CLOSES
        generation_duration = time.time() - inference_start_time
        ttft = (first_token_time - inference_start_time) if first_token_time else 0
        
        # 4. Token & Telemetry Extraction (Updated for Stochastic Data)
        completion_tokens = len(full_text.split()) 
        
        # Pull arrays from our time-series log for dynamic evaluation
        tps_samples = [s['instantaneous_tps'] for s in timeseries_log] if timeseries_log else [0]


        import numpy as np 
        
        tps = float(np.mean(tps_samples)) 

        itl = (generation_duration / total_tokens_generated) * 1000 if total_tokens_generated > 0 else 0
        
        # Tokenization metrics
        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
        completion_tokens = len(llm.tokenize(full_text.encode("utf-8")))
        context_load_ratio = prompt_tokens / completion_tokens if completion_tokens > 0 else 0

        
        # print("\n📈 --- LIVE INFRASTRUCTURE METRICS REPORT (STOCHASTIC) ---")
        print(f"⏱️ Generation Duration: {generation_duration:.2f} seconds")
        print("-------------------------------------------\n")
        
        print("\n⚡ --- INFERENCE PERFORMANCE METRICS ---")
        print(f"⏱️ TTFT (Prefill):     {ttft:.4f} seconds")
        print(f"🚀 Avg TPS (Decoding):   {tps:.2f} tokens/sec")
        print(f"⏳ ITL (Avg Latency):  {itl:.2f} ms/token")
        print(f"📊 Prompt-to-Output Ratio: {context_load_ratio:.2f}")
      

        # --- PREPARE METADATA ROW ---
        # Capture the specific configuration state
        model_name = params.get("model", "unknown")
        n_gpu_layers = params.get("num_gpu", -1) 
        n_ctx = params.get("num_ctx", 2048)
        f16_kv = params.get("f16_kv", True)

        metrics_log = {
            "RUN_ID": RUN_ID,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            
            # --- ENVIRONMENT & ARCHITECTURE ---
            "runtime_engine": "llama-cpp-python",
            "backend": "CUDA 13.3",
            "framework": "GGUF",

            # --- SPECULATIVE DECODING ---
            "speculative_decoding": params.get("speculative_enabled", False),
            "draft_model": params.get("draft_model", "None"),

            # --- GPU & VRAM SWEEP ---
            "gpu_util_target": params.get("gpu_util_target", 0.70), # The target sweep value
            # --- KV CACHE & PRECISION ---
            "kv_cache_precision": params.get("kv_cache_type", "FP16"),
            "n_ctx": params.get("num_ctx", 4096),

            # --- MODEL & RUNTIME VARS ---
            "model_name": model_name,
            "n_ctx": n_ctx,
            "num_gpu_layers": n_gpu_layers,
            "quantization": params.get("quantization"),
            "f16_kv": f16_kv,

           # --- TELEMETRY METRICS (STOCHASTIC OVER VIEW) ---
            "timeseries_history": timeseries_log, 
            "ttft_sec": round(ttft, 4),
            "tps": round(tps, 2),
            "itl_ms": round(itl, 2),
            "prompt_to_output_ratio": round(context_load_ratio, 2),
            "latency": generation_duration
        }

        print("generation_duration:", generation_duration)

        raw_text = full_text


        return raw_text, metrics_log
        
    except Exception as e:
        print(f"⚠️ Local parser execution anomaly: {e}")
        try:
            print("==========ERRTORR==================\n")
        except:
            pass
        # Return a structured error fallback so the main loop triggers a 1-hour cooldown sleep
    
        return "Error 200002", {}



if __name__ == "__main__":

    
    
    start = datetime.now()
    
    # 1. Load Configurations globally at startup
    CONFIG_FILE = "config.json"
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config_data = json.load(f)
        print("⚙️ Successfully loaded keyword configurations from JSON layout.")

    params = config_data.get("LLAMA_CPP_PARAMS", {})
    target_temperature = config_data.get("TARGET_TEMP", 50)
    print(params)
    print("TARGET_TEMP:", target_temperature)
    # json_schema = config_data.get("JSON_SCHEMA", {})


    # 🧊 =========================================================
    # 🔥 COLD ENGINE START: INITIAL ENVIRONMENT STABILIZATION
    # =========================================================
    print("\n🥶 [INITIALIZATION] Ensuring clean environment baseline before loop entry...")
    cool_down(target_temp=target_temperature) 
    print("✅ System confirmed cold. Entering sweep matrix layout.\n")

    # 📊 DEFINE THE MODELS MATRIX TO LOOP THROUGH 
    benchmark_models = [
        {
            "repo_id": "unsloth/Qwen2.5-VL-3B-Instruct-GGUF",
            "filename": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
            "model_chat_format": "chatml",
            "label": "Qwen2.5-VL-3B"
        },
        {
            "repo_id": "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
            "filename": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 
            "model_chat_format": "chatml",
            "label": "Qwen2.5-VL-7B"
        }
    ]


    # 3. Process Execution Matrix 
    # ctx_sizes = [512, 2048, 4096, 8192]
    # gpu_layers = [-1, 20, 36]
    
    ctx_sizes = [4096]
    gpu_layers = [-1]


    

    for model_meta in benchmark_models:
        print("\n" + "=============================================================")
        print(f"📦 [MODEL CORE SWITCH] Activating weights for: {model_meta['label']}")
        print("=============================================================")

        
        # Inject the active model meta parameters dynamically into the params dict
        params["model_repo"] = model_meta["repo_id"]
        params["model_filename"] = model_meta["filename"]
        params["model"] = model_meta["label"]  # Tracks it nicely inside CSV log output
        params["model_chat_format"] = model_meta["model_chat_format"]

        for ctx in ctx_sizes:
            for layers in gpu_layers:
                print(f"\n🚀 [MATRIX] Initializing Sweep Configuration -> Context: {ctx} | GPU Layers: {layers}")

                # Generate a unique Version 4 (random) UUID for this run
                RUN_ID = str(uuid.uuid4())
                print(f"Starting script execution. Run ID: {RUN_ID}")

                print("📺 Starting Live Parallel Telemetry Monitor...")
                print("ℹ️ Continuous monitoring mode active. No services will be closed or altered.\n")
    
                # Initialize CSV destination
                initialize_csv()

                base_fans = get_instantaneous_fans()
                print(f"🌬️ Baseline Fans (Before Running): {base_fans}")
                
                # log_to_csv_thermal(RUN_ID)

                # ─── THREAD MANAGEMENT START ───
                # Create a signaling flag to stop the thread later
                stop_logger_event = threading.Event()

                # Initialize the background thread
                logger_thread = threading.Thread(
                    target=continuous_thermal_logger, 
                    args=(RUN_ID, stop_logger_event),
                    daemon=True 
                )

                # Start logging every second in the background
                logger_thread.start()

                # Update parameters dynamically for this specific run
                params["num_ctx"] = ctx
                params["num_gpu"] = layers
                
                # --- RUN BENCHMARK ---
                res, logs = fast_extract_table(params, RUN_ID)

                # ─── STOP THE LOGGER ───
                # The benchmark finished, so signal the thread to stop and wait for it to clean up
                stop_logger_event.set()
                logger_thread.join()

                # ─── CHECK FOR CUDA/OOM FALLBACK ERROR ───
                if res == "Error 200002":
                    print(f"⚠️ [OOM / CUDA SKIP] Configuration Ctx:{ctx}/Lyrs:{layers} failed allocation. Skipping safely...")
                    print("-------------------------------------------------------------\n")
                    continue # Jumps directly to the next iteration in the sweep matrix

                # Write summary out to database log
                if logs:
                    log_to_csv(logs)

                # 🧊 =========================================================
                # 🔥 POST-RUN RECOVERY MIGRATION MATRIX
                # =========================================================
                print(f"\n❄️ Benchmark complete for Ctx:{ctx}/Lyrs:{layers}. Invoking recovery sweep...")
                
                # 1. Clears driver state, locks low clocks, and drains core heat down to 45C
                cool_down(target_temp=target_temperature)
                
                # 2. Enforce structural stabilization window
                # Holds absolute idle to completely dissipate heat evenly across VRAM pads
                STABILIZATION_TIMEOUT_SEC = 300  # 5 Minutes (set to 600 if you want 10)
                print(f"⏳ Verification: Enforcing {STABILIZATION_TIMEOUT_SEC/60:.1f} min stabilization window...")
                time.sleep(STABILIZATION_TIMEOUT_SEC)
                
                print("-------------------------------------------------------------\n")
        
        gc.collect()
        time.sleep(5)

    print(f"🏁 Full sweeping matrix finalized. Combined runtime: {datetime.now() - start}")

