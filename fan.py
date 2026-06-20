# fan_speed.py

import os
import pynvml 
import time
import gc
import sys
import subprocess
import datetime
import psutil
import csv
import re

HWINFO_LOG_FILE = r"C:\Users\AbdulAhmed\Downloads\CPU vs GPU\hwdata.CSV"
OUTPUT_CSV_FILE = r"C:\Users\AbdulAhmed\Downloads\CPU vs GPU\telemetry_log.csv"

# --- INITIALIZATION ---
try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_GPU_TRACKING = True
except Exception as e:
    print(f"⚠️ NVML Initialization Failed: {e}. Running without deep telemetry.")
    HAS_GPU_TRACKING = False


def initialize_csv():
    """Creates the CSV file and writes headers if it doesn't exist."""
    if not os.path.exists(OUTPUT_CSV_FILE):
        headers = [
            "RUN_ID", "Timestamp", "PState", "VRAM_GB", "GPU_Util_Pct", "CPU_Load_Pct", 
            "RAM_Used_GB", "RAM_Util_Pct", "GPU_Mem_Util_Pct", "Power_W", 
            "Temp_C", "Clock_MHz", "Clock_Deficit","CPU_Fan_Speed", "GPU_Fan_Speed"
        ]
        try:
            os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
            with open(OUTPUT_CSV_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"📝 Created new CSV log file at: {OUTPUT_CSV_FILE}")
        except Exception as e:
            print(f"⚠️ Failed to initialize CSV file: {e}")


def get_ram_telemetry_data():
    """Returns raw RAM metrics for the CSV file structure."""
    mem_info = psutil.virtual_memory()
    return mem_info.used / (1024**3), mem_info.percent

def continuous_thermal_logger(exp_name, stop_event):
    """
    Runs in a background thread, writing metrics to a CSV every 1 second 
    until the stop_event is set.
    """
    print(f"📊 Background telemetry logger started for Run ID: {exp_name}")
    
    while not stop_event.is_set():
        try:
            metrics = get_gpu_silicon_telemetry()
            cpu_load = get_cpu_utilization()
            ram_gb, ram_pct = get_ram_telemetry_data()
            MAX_BOOST_CLOCK = 2100
            clock_stochastic = metrics["clock_mhz"]
            clock_deficit_prec = round(max(0, ((MAX_BOOST_CLOCK - clock_stochastic) / MAX_BOOST_CLOCK) * 100))

            fan_string = get_instantaneous_fans() 
            cpu_fan_val = ""
            gpu_fan_val = ""
            
            if fan_string and "Not Found" not in fan_string and "Gathering" not in fan_string:
                parts = fan_string.split("|")
                for part in parts:
                    part_lower = part.lower()
                    nums = re.findall(r'\d+', part)
                    if nums:
                        if "cpu" in part_lower:
                            cpu_fan_val = nums[0]
                        elif "gpu" in part_lower:
                            gpu_fan_val = nums[0]

            # Match the main script's standard datetime usage
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            row = [
                exp_name,
                timestamp,
                f"P{metrics.get('pstate', 'N/A')}",
                f"{metrics['vram_gb']:.2f}",
                metrics['gpu_util'],
                cpu_load,
                f"{ram_gb:.2f}",
                ram_pct,
                metrics['mem_util'],
                f"{metrics['power_w']:.1f}",
                metrics['temp_c'],
                metrics['clock_mhz'],
                clock_deficit_prec,
                cpu_fan_val,
                gpu_fan_val
            ]

            with open(OUTPUT_CSV_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            print(f"⚠️ Error writing to CSV: {e}")
            
        # Precise 1-second interval wait, but wakes up instantly if stop_event is set
        stop_event.wait(timeout=1.0)
        
    print(f"🛑 Background telemetry logger stopped for Run ID: {exp_name}")

def log_to_csv_thermal(exp_name):
    """Gathers current metrics and logs them with complete time and numeric fan data."""
    metrics = get_gpu_silicon_telemetry()
    cpu_load = get_cpu_utilization()
    ram_gb, ram_pct = get_ram_telemetry_data()
    MAX_BOOST_CLOCK = 2100
    clock_stochastic = metrics["clock_mhz"]
    clock_deficit_prec = round(max(0, ((MAX_BOOST_CLOCK - clock_stochastic) / MAX_BOOST_CLOCK) * 100))

    
    # 1. Get the raw fan string from the existing untouched function
    # e.g., "CPU Fan [RPM]: 3400 | GPU Fan [RPM]: 3700"
    fan_string = get_instantaneous_fans() 
    
    cpu_fan_val = ""
    gpu_fan_val = ""
    
    # 2. Parse the numbers out based on 'cpu' and 'gpu' keywords found in the string
    if fan_string and "Not Found" not in fan_string and "Gathering" not in fan_string:
        parts = fan_string.split("|")
        for part in parts:
            part_lower = part.lower()
            nums = re.findall(r'\d+', part)
            if nums:
                if "cpu" in part_lower:
                    cpu_fan_val = nums[0]
                elif "gpu" in part_lower:
                    gpu_fan_val = nums[0]

    # 3. FIXED: Added %S for explicit second capture
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [
        exp_name,
        timestamp,
        f"P{metrics.get('pstate', 'N/A')}",
        f"{metrics['vram_gb']:.2f}",
        metrics['gpu_util'],
        cpu_load,
        f"{ram_gb:.2f}",
        ram_pct,
        metrics['mem_util'],
        f"{metrics['power_w']:.1f}",
        metrics['temp_c'],
        metrics['clock_mhz'],
        clock_deficit_prec,
        cpu_fan_val,  # Pure number column
        gpu_fan_val   # Pure number column
    ]

    try:
        with open(OUTPUT_CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print(f"⚠️ Error writing to CSV: {e}")


def get_instantaneous_fans():
    if not os.path.exists(HWINFO_LOG_FILE):
        return "HWiNFO Log Not Found"
    try:
        with open(HWINFO_LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return "Gathering data..."

        # DEBUG: Print the headers to the console so we can see what to look for
        headers = [h.strip().replace('"', '') for h in lines[0].split(",")]
        # print(f"DEBUG: Found headers in CSV: {headers}") 

        live_values = [v.strip().replace('"', '') for v in lines[-1].split(",")]
        fan_metrics = []
        for idx, header in enumerate(headers):
            # Let's be less strict for debugging
            if "fan" in header.lower() or "rpm" in header.lower():
                fan_metrics.append(f"{header}: {live_values[idx]}")
                    
        return " | ".join(fan_metrics) if fan_metrics else "No columns matching 'fan' or 'rpm' found."
    except Exception as e:
        return f"Error: {e}"
    

    
def get_gpu_silicon_telemetry():
    """Queries raw silicon metrics directly via NVIDIA kernel driver."""
    if not HAS_GPU_TRACKING:
        return {
            "vram_gb": 0.0,
            "temp_c": 0,
            "clock_mhz": 0,
            "gpu_util": 0,
            "mem_util": 0,
            "power_w": 0.0,
        }

    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    temp = pynvml.nvmlDeviceGetTemperature(
        gpu_handle,
        pynvml.NVML_TEMPERATURE_GPU
    )
    clock = pynvml.nvmlDeviceGetClockInfo(
        gpu_handle,
        pynvml.NVML_CLOCK_GRAPHICS
    )

    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
    power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0
    pstate = pynvml.nvmlDeviceGetPerformanceState(gpu_handle)

    return {
        "vram_gb": mem.used / (1024**3),
        "temp_c": temp,
        "clock_mhz": clock,
        "gpu_util": util.gpu,      # GPU utilization %
        "mem_util": util.memory,   # Memory controller utilization %
        "power_w": power,
        "pstate": pstate,
    }




def get_cpu_utilization():
    """Queries system-wide CPU utilization percentage."""
    # interval=None provides a non-blocking check based on the last call or import slice
    return psutil.cpu_percent(interval=None)

def get_ram_telemetry():
    """Queries system-wide Host RAM utilization."""
    mem_info = psutil.virtual_memory()
    # Convert bytes used to Gigabytes
    used_gb = mem_info.used / (1024**3)
    percent = mem_info.percent
    return f"RAM: {used_gb:.2f} GB ({percent}%)"

def print_live_telemetry_line():
    """Helper to cleanly print the exact uniform telemetry status line."""
    metrics = get_gpu_silicon_telemetry()
    cpu_load = get_cpu_utilization()
    ram_usage = get_ram_telemetry()
    # CHANGE THIS LINE: Add the extra .datetime so it matches the top import
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    MAX_BOOST_CLOCK = 2100
    clock_stochastic = metrics["clock_mhz"]
    clock_deficit_prec = round(max(0, ((MAX_BOOST_CLOCK - clock_stochastic) / MAX_BOOST_CLOCK) * 100))


    print(
        f"[{timestamp}] "
        f"P{metrics.get('pstate', 'N/A')} | "
        f"VRAM: {metrics['vram_gb']:.2f} GB | "
        f"GPU Util: {metrics['gpu_util']}% | "
        f"CPU Load: {cpu_load}% | "
        f"{ram_usage} | "  # <-- REMOVED THE TRAILING COMMA HERE
    )
    print(
        f"Memory: {metrics['mem_util']}% | "
        f"Power: {metrics['power_w']:.1f} W | "
        f"TEMP: {metrics['temp_c']}°C | "
        f"CLOCK: {metrics['clock_mhz']} MHz | "
        f"CLOCK DEFECIT: {clock_deficit_prec}% | ",  # Added a clean comma separation here before arguments
        flush=True
    )

    print(get_instantaneous_fans(), flush=True)
    
# --- Execution ---
if __name__ == "__main__":
    print("📺 Starting Live Parallel Telemetry Monitor...")
    print("ℹ️ Continuous monitoring mode active. No services will be closed or altered.\n")
    
    # Initialize CSV destination
    # initialize_csv()

    base_fans = get_instantaneous_fans()
    print(f"🌬️ Baseline Fans (Before Running): {base_fans}")
    
    # Simple formatting separator
    print("-" * 60)
    
    try:
        while True:
            # log_to_csv_thermal("Fri_10_00")

            # Reuses the exact uniform telemetry line function perfectly
            print_live_telemetry_line()
            print("-" * 60)
            
            # Refresh interval in seconds (change to 5 or 10 if  want faster updates)
            time.sleep(5) 
            
    except KeyboardInterrupt:
        print("\n🛑 Telemetry monitor stopped by user.")
