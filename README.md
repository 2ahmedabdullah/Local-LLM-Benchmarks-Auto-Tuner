# Empirical Benchmarking of Local LLM Inference Under Thermal Constraints (Using Cooling Pads)

This repository evaluates local Large Language Model (LLM) inference performance under strict hardware constraints, mapping the relationship between software configurations and real-time hardware telemetry. The framework systematically benchmarks how variables like GPU layer offloading, context window depth, and quantization matrix bit-widths affect Tokens Per Second (TPS), Time to First Token (TTFT), and VRAM allocation.

Beyond standard software profiling, this suite features a controlled silicon thermal study. By benchmarking identical model configurations across standardized starting temperatures ($<50^\circ\text{C}$, $60^\circ\text{C}$, $70^\circ\text{C}$, and $80^\circ\text{C}$), the project empirically demonstrates the physical limits of edge hardware—proving that high-temperature environments trigger dynamic thermal throttling that degrades generation throughput by nearly 10% compared to a stabilized cold baseline.


## ⚙️ What it tests

The system runs structured experiments across a multi-dimensional parameter grid:

Context Sizes: (e.g., 512, 2048, 4096, 8192 tokens)

GPU Layer Offloading Levels: Granular CPU/GPU split ratios

Model Quantization Formats: (e.g., Q4_K_M, Q5_K_M)

Speculative Decoding Settings: Optional draft-model configurations

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


Below is the combined performance metrics log capturing throughput alongside system hardware telemetry:

| Timestamp | Instantaneous TPS | TPS | ITL | VRAM (GB) | RAM Used (GB) | Temp (°C) | Clock Deficit (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-06-19 20:00:50 | 40.2 | 48.8 | 20.5 | 2.6 | 3.6 | 66.0 | 17.0 |
| 2026-06-19 20:00:51 | 40.5 | 48.8 | 20.5 | 2.6 | 3.6 | 66.0 | 17.0 |

## Time Series Data

Skip time-series logging in the prefill phase because the mathematical time interval is too small and unstable. Prefill (TTFT) usually takes anywhere from $20\text{ms}$ to $300\text{ms}$ ($0.02$ to $0.3$ seconds). Because time-series window is configured to sample every $1$ seconds, the entire prefill phase finishes way before the very first $0.5$-second telemetry bucket can even close.


Timeline
 
Phase 1: Pipeline Start (Model Loading)
What happens: When llm = Llama(...) runs, the entire GGUF model weights are read from the disk and immediately packed into the GPU layers.

Phase 2: Prefill Phase (Inference Launch / Prompt Processing)
What happens: The model weights are already sitting in VRAM. When prompt is passed, the GPU processes all the input tokens simultaneously to build the initial Key-Value (KV) Cache. This adds a tiny bit of extra VRAM usage for the context cache.

Phase 3: Decoding Phase (Token Generation Streaming)
What happens: As each new token is generated one by one, it is appended to the KV cache, causing VRAM to creep up slightly until generation finishes.

Phase 4: Cooling using Eye pads 
By introducing the driving cooldown matrix (cool_down(target_temp=50)) and the structural stabilization window at the end of the run, one guarantee that Run #2 isn't penalized by the residual heat of Run #1. This isolates the variables completely, meaning changes in the parameters (like n_gpu_layers or kv_cache_precision) will show their true impact on performance, completely free of thermal bias.

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


──────────────────────┐
                    
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



CPU and GPU operate as independent compute units.

Both dissipate heat into a shared copper heat pipe system.

Thermal equilibrium is determined by total system power draw.

Cooling system (fans + fins) responds to combined heat load.

Throttling occurs when any sensor (CPU or GPU) exceeds thermal limits.


## 🌡️ Thermal considerations

Local LLM performance is affected by thermal throttling, especially on laptops.

To reduce noise in benchmark results, runs are executed under controlled temperature conditions using a cooldown step before each test.

Fan and temperature data is optionally collected using HWiNFO logs on Windows systems.



## 🧊 Thermal Management & Cooling

Performance in local LLM testing is limited by how well a laptop handles heat. On laptops where the CPU and GPU share the same cooling system, temperatures rise quickly during intensive tasks. In high ambient temperatures (e.g., $32^\circ\text{C}$ in Pune), this leads to thermal throttling, which slows down token generation and can invalidate benchmark results.

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


                ┌────────────────────┐
                │  Config Sweep Loop │
                │ (ctx, layers, etc) │
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


## 📁 Output

Each run generates a structured log containing:

model configuration
hardware state
performance metrics
thermal conditions


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
  0    50          80                        100.0  GPU Surface Temp (Celcius)


## ⚠️ Notes

This project is for experimental benchmarking and performance analysis of local inference systems. Results may vary depending on hardware, drivers, and system load.




