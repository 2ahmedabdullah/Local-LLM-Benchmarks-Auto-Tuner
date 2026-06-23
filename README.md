# Empirical Benchmarking of Local LLM Inference Under Thermal Constraints (Using Cooling Pads)

This repository evaluates local Large Language Model (LLM) inference performance under strict hardware constraints, mapping the relationship between software configurations and real-time hardware telemetry. The framework systematically benchmarks how variables like context window depth, and quantization matrix bit-widths affect Tokens Per Second (TPS), Time to First Token (TTFT), and VRAM allocation.

Beyond standard software profiling, this suite features a controlled silicon thermal study. By benchmarking identical model configurations across standardized starting temperatures ($<50^\circ\text{C}$, $60^\circ\text{C}$, $70^\circ\text{C}$, and $80^\circ\text{C}$), the project empirically demonstrates the physical limits of edge hardware—proving that degrades generation throughput by nearly 10% compared to a stabilized cold baseline. Performance ceiling is constrained by RTX 3050-class laptop 6 GB GPU VRAM.


## ⚙️ What it tests

The system runs structured experiments across a multi-dimensional parameter grid:

Model: Qwen2.5-VL-3B

Context Sizes: ctx_sizes = [65536, 32768, 16384, 8192, 4096]

Model Quantization Formats: [F16, Q8_0, Q4_K_M]

GPU Layer Offloading Levels: [-1] -> 100% GPU Only

All combinations are repeated twice at different Silicon Temp Start = [50C, 60C, 70C, 80C]

So for 50C Temp, there are 15 such combinations = Total 60 experiments.

While evaluating each configuration using a standardized prompt set, the framework systematically varies the baseline silicon starting temperature. During execution, an asynchronous telemetry thread captures instantaneous Tokens Per Second (TPS) alongside real-time core temperatures, mapping exactly how dynamic thermal loading impacts active generation throughput.


## 📊 Metrics collected

For each run, the following are recorded:

Instantaneous Tokens per second (TPS)
Time to first token (TTFT)
Inter-token latency (ITL)
GPU temperature
VRAM usage
RAM Usage
Estimated throttling behavior


### 📋 Full Telemetry Data


| Timestamp | PState | VRAM (GB) | GPU Util (%) | CPU Load (%) | RAM Used (GB) | RAM Util (%) | GPU Mem Util (%) | Power (W) | Temp (°C) | Clock (MHz) | Clock Deficit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 21-06-2026 10:46:36 | P0 | 0.14 | 0 | 0.0 | 15.16 | 63.9 | 0 | 8.9 | 41 | 1492 | 29 |
| 21-06-2026 10:46:37 | P0 | 0.14 | 0 | 10.2 | 14.97 | 63.1 | 0 | 5.5 | 40 | 1492 | 29 |
| 21-06-2026 10:46:38 | P0 | 0.20 | 0 | 16.3 | 15.08 | 63.6 | 0 | 8.7 | 42 | 1492 | 29 |
| 21-06-2026 10:46:40 | P0 | 5.27 | 9 | 36.2 | 15.37 | 64.8 | 1 | 11.4 | 42 | 1845 | 12 |
| 21-06-2026 10:46:41 | P0 | 5.92 | 59 | 34.4 | 16.29 | 68.7 | 18 | 15.4 | 43 | 1965 | 6 |


> **Note:** All rows map back to `RUN_ID: a9506584-709b-47ac-974f-1f84949567c7`



| Second | Runtime Engine | Backend | Framework | KV Cache | Context ($n\_ctx$) | Model | GPU Layers | F16 KV | TTFT (s) | Avg TPS | ITL (ms) | P:O Ratio | Latency (s) | Instant TPS |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | llama-cpp-python | CUDA 13.3 | GGUF | bf16 | 98304 | Qwen2.5-VL-3B-Q8 | -1 | TRUE | 2.89 | 31.77 | 32.1 | 0.21 | 131.47 | 41.47 |
| 2 | llama-cpp-python | CUDA 13.3 | GGUF | bf16 | 98304 | Qwen2.5-VL-3B-Q8 | -1 | TRUE | 2.89 | 31.77 | 32.1 | 0.21 | 131.47 | 40.22 |
| 3 | llama-cpp-python | CUDA 13.3 | GGUF | bf16 | 98304 | Qwen2.5-VL-3B-Q8 | -1 | TRUE | 2.89 | 31.77 | 32.1 | 0.21 | 131.47 | 40.09 |
| 4 | llama-cpp-python | CUDA 13.3 | GGUF | bf16 | 98304 | Qwen2.5-VL-3B-Q8 | -1 | TRUE | 2.89 | 31.77 | 32.1 | 0.21 | 131.47 | 39.81 |
| 5 | llama-cpp-python | CUDA 13.3 | GGUF | bf16 | 98304 | Qwen2.5-VL-3B-Q8 | -1 | TRUE | 2.89 | 31.77 | 32.1 | 0.21 | 131.47 | 38.59 |


## Time Series Data

Skip time-series logging in the prefill phase because the mathematical time interval is too small and unstable. Prefill (TTFT) usually takes anywhere from $20\text{ms}$ to $300\text{ms}$ ($0.02$ to $0.3$ seconds). Because time-series window is configured to sample every $1$ seconds, the entire prefill phase finishes way before the very first $0.5$-second telemetry bucket can even close.


![Time Series Data](system_performance_plot.png)

```text
1) The system is GPU-bound but stable after warm-up: VRAM line (~3 GB range) quickly ramps up and then stays flat
  → means the model/load is allocated once and reused efficiently
  → no memory leaks or repeated reloading


2) Throughput (TPS) has a classic “startup spike → steady state → decay” pattern. TPS jumps to ~70 early then settles around ~40–45 tokens/sec
  Interpretation:
  Initial burst = model warm-up / cache priming / kernel optimization
  Drop afterward = real sustained inference throughput
  Key insight: It’s fast initially, then stabilizes at realistic sustained throughput.


3) Inter-token latency (ITL) is flat after stabilization: ITL becomes almost constant (~20-ish)
  This implies: stable decoding loop, no jitter from memory swapping, no OS scheduling instability
  hence -> predictable latency > peak speed in production systems.


4) Temperature rises then plateaus (no runaway thermal curve) temp climbs early (load phase) then flattens instead of continuously rising
This tells us: cooling system is coping no thermal throttling spiral. If there were throttling, one’d see: TPS collapsing while temp keeps rising


5) Clock deficit + VRAM behavior suggests mild throttling, not catastrophic throttling clock deficit fluctuates but doesn’t explode GPU clock likely reduces slightly under sustained load.


6) RAM is irrelevant in this pipeline: “Scaled RAM” line is basically flat → This confirms workload is GPU-contained, not CPU-swapping.
```

## Timeline
 
Phase 1: Pipeline Start (Model Loading)
What happens: When llm = Llama(...) runs, the entire GGUF model weights are read from the disk and immediately packed into the GPU layers.

Phase 2: Prefill Phase (Inference Launch / Prompt Processing)
What happens: The model weights are already sitting in VRAM. When prompt is passed, the GPU processes all the input tokens simultaneously to build the initial Key-Value (KV) Cache. This adds a tiny bit of extra VRAM usage for the context cache.

Phase 3: Decoding Phase (Token Generation Streaming)
What happens: As each new token is generated one by one, it is appended to the KV cache, causing VRAM to creep up slightly until generation finishes.

Phase 4: Cooling using Eye pads 
By introducing the driving cooldown matrix (cool_down(target_temp=50)) and the structural stabilization window at the end of the run, one guarantee that Run #2 isn't penalized by the residual heat of Run #1 too isolates the variables.

Phase 5: Stabilizing Time = 5 min



## ⚙️ Target Hardware (The Constraints)

This suite is deliberately designed to profile budget, edge, and consumer-grade hardware configurations:

```toml
[Hardware Profile]
Device       = "RTX 3060 Laptop (6GB VRAM) / RTX 4050 (6GB VRAM)"
Compute      = "Intel i7-12700H / Ryzen 7"
System RAM   = "16GB / 32GB DDR4/DDR5"
OS           = "Windows 11"
```


## Lenovo LOQ Cooling Inbuilt Design

The Lenovo LOQ 15IRX9 (Type 83DV)—uses Lenovo’s updated hyperchamber cooling design (Dual Super Falcon fans). Physically, it relies on a shared copper thermal block and overlapping heat pipes bridging the Intel i7-13650HX CPU and the NVIDIA GPU.

```text
────────────────────────────────────┐               
                     ┌──────────────┼──────────────┐
                     │                             │
                     │     Dual Super Falcon Fans  │
                     │  (Lenovo Hyperchamber Exit) │
                     │                             │
                     └──────────────▲──────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            │            Radiant Heat Extraction            │
            │                       │                       │
    ┌───────┴────────┐      ┌───────┴────────┐     ┌────────┴───────┐
    │  Intel i7 CPU  │      │ Overlapping    │     │   NVIDIA GPU   │
    │   13650HX      │      │ Copper Pipes   │     │(3050/4050/4060)│
    │                │      │                │     │                │
    │  Tokenization  │      │ Shared Thermal │     │ Tensor Matrix  │
    │  & Prompt      │      │ Saturation Pool│     │ Compute Loop   │
    │  Coordination  │      │                │     │ (Kernel Exec)  │
    └───────▲────────┘      └───────▲────────┘     └────────▲───────┘
            │                       │                       │
            └──────────────┬────────┴─────────┬─────────────┘
                           │                  │
              [CPU Burst Power spikes]     [Sustained GPU TGP]
                        (TTFT Phase)         (Sustained TPS)

```

CPU and GPU operate as independent compute units.

Both dissipate heat into a shared copper heat pipe system.

Thermal equilibrium is determined by total system power draw.

Cooling system (fans + fins) responds to combined heat load.


## 🌡️ Thermal considerations

To reduce noise in benchmark results, runs are executed under controlled temperature conditions using a cooldown step before each test.

Fan and temperature data is optionally collected using HWiNFO logs on Windows systems.



## 🧊 Thermal Management & Cooling

Performance in local LLM testing is limited by how well a laptop handles heat. On laptops where the CPU and GPU share the same cooling system, temperatures rise quickly during intensive tasks. In high ambient temperatures (e.g., $32^\circ\text{C}$ in Pune), which can slow down token generation and can invalidate benchmark results.

To maintain stable performance and keep temperatures below $50^\circ\text{C}$ during testing, the author uses a cold pack (an eye-gel ice mask wrapped in a thin sock).Placement: The pack is placed under the laptop chassis, away from the intake vents, to act as a heat sink for the base of the device.Safety: The sock prevents condensation from entering the laptop vents or damaging the internal components.

To maintain a standardized environment, the author utilizes the following workflow:Initial Baseline: The cold pack is applied to ensure every benchmark run initiates at a common, stable temperature below $50^\circ\text{C}$.Active Management: Once the target starting temperature is achieved, the cooling pad is removed before the benchmark sequence begins. This ensures that the testing environment remains consistent and reproducible across multiple runs.

Followed by a "Warm-Up" Period of fixed 10 mins without extra cooling to allow the system to reach thermal equilibrium before recording any data or running new models.




## 🔌 Beyond NVML: Hardware Telemetry via HWiNFO64

NVIDIA's official management library (pynvml / NVML) provides native APIs for desktop graphics cards to check fan speeds with a single line of code: nvmlDeviceGetFanSpeed().

However, on laptop form-factors, this API fails. Laptop GPU fans are not controlled by the graphics card itself; they are managed by a dedicated chip on the motherboard called the Embedded Controller (EC). The EC handles the system's thermal profiles dynamically, balancing heat dissipation between the CPU and GPU. Because every manufacturer (ASUS, Lenovo, Dell, HP) deploys proprietary, closed-source EC firmware, there is no universal API to read laptop fan RPMs directly through Python.

For inferencing, the framework requires granular data on how laptop cooling systems adapt to sustained processing loads. Because the EC restricts this telemetry from the standard NVIDIA driver, the system implements an asynchronous bridge using HWiNFO64.


⚠️ Note: Users must ensure the HWiNFO CSV logging feature is enabled and its directory path is correctly pointed to within config.json to utilize this telemetry loop


## 🧪 Experimental design

The system performs grid-style benchmarking across multiple configuration combinations and logs results to CSV for later analysis.


## 📊 Benchmark dataset definition


Input dataset: 20 fixed prompts

mixed categories:
short reasoning
long context
code generation
chat-style prompts

Constraints:
same prompts across all runs
same temperature (e.g. 0.7)
same max_tokens (e.g. 512)

For each run:

(config) → (metrics vector)

Example:
```
{
  "ctx": 4096,
  "gpu_layers": 36,
  "quant": "Q4_K_M",
  "tps": 18.7,
  "ttft": 0.62,
  "vram": 5.2
}
```

## Baseline comparisons 

Baselines:
A. CPU-only inference
n_gpu_layers = 0

B. Full GPU offload
n_gpu_layers = -1

C. Default config
llama.cpp default settings

D. Quant baseline
Q8 vs Q4 comparison



## 🧱 Architecture Diagram


```text
                ┌────────────────────┐
                │  Config Sweep Loop │
                │ (ctx, quantization)│
                └─────────┬──────────┘
                          │
                          ▼
        ┌────────────────────────────────┐
        │   llama-cpp-python Inference   │
        │  (GGUF models + settings)      │
        └─────────┬──────────────────────┘
                  │ streaming output
                  ▼
     ┌──────────────────────────────┐
     │  Benchmark Metrics Collector │
     │ - TPS / TTFT / ITL           │
     │ - VRAM / clocks / temp       │
     └─────────┬────────────────────┘
               │
               ▼
   ┌──────────────────────────┐
   │ NVML + HWiNFO Telemetry  │
   │ GPU + fan + power data   │
   └─────────┬────────────────┘
             ▼
   ┌──────────────────────────┐
   │ CSV / JSON Experiment DB │
   └──────────────────────────┘
```

## 📁 Output

Each run generates a structured log containing:

model configuration
hardware state
performance metrics
thermal conditions


```text
Inference Speed (TPS)
  ▲ 
  │          [Optimal Points]
  │                 ★ (Fastest / Low Context)
  │                /
  │               ★ (The Sweet Spot)
  │              /
  │             ★ (Slower / Massive Context)
  │            / 
  │           /   ● (Safe but slow config)
  │          /
  │         /        ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
  │        /         ■  CRASH ZONE (OOM)       ■
  │       /          ■  (VRAM > 4.2 GB)        ■
  │      /           ■                         ■  
  └─────┴────────────┴─────────────────────────┴─────►
  0    1.0          4.2 (Safety Ceiling)      6.0  VRAM Used (GB)

```

```text

Inference Speed (TPS)
  ▲
  │          [Optimal Points]
  │                 ★ (Fastest / Low Context)
  │                /
  │               ★ (The Sweet Spot)
  │              /
  │             ★ (Slower / Massive Context)
  │            / 
  │           /   ● (Safe but slow config)
  │          /
  │         /        ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■
  │        /         ■  HIGH TEMP ZONE         ■
  │       /          ■                         ■
  │      /           ■                         ■
  └─────┴────────────┴─────────────────────────┴─────►
  0    50           70                        100.0  GPU Surface Temp (Celcius)

```

## ⚠️ Notes

This project is for experimental benchmarking and performance analysis of local inference systems. Results may vary depending on hardware, drivers, and system load.




