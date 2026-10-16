# Table 9: Edge Device Inference Benchmark (**SIMULATED**)

| System Characteristic | High-Precision Pipeline (FP32) | Edge Quantized Pipeline (INT8) | Relative Optimization |
| :--- | :--- | :--- | :--- |
| **Binary Model Footprint** | 89.2 MB | **22.8 MB** | 74.4% storage compression |
| **Peak Operational Memory** | 142.0 MB | **48.5 MB** | 65.8% RAM reduction |
| **Turn Processing Latency (Median)** | 48.2 ms | **18.4 ms** | 61.8% latency speedup |
| **Turn Processing Latency (95th Pct)**| 65.0 ms | **24.2 ms** | 62.8% tail latency speedup |
| **Throughput (Turns / Second)** | 20.7 | **54.3** | $2.6\times$ throughput |
| **Estimated Energy / 5-min Check-in**| 0.12% battery | **0.04% battery** | 66.7% energy reduction |

*Status Label: **SIMULATED**. Execution measured on simulated mobile ARM runtime environment. No student personal hardware was accessed during benchmarking.*
