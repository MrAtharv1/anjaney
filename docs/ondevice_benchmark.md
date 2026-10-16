# ANJANEY — On-Device Inference Benchmark (Simulated)

> ⚠️ **SIMULATION NOTICE**: Measurements captured in a local environment simulating mobile CPU constraints.

- **Peak Host Memory (RSS):** 93.3 MB
- **Model Architecture:** `all-MiniLM-L6-v2` (6-layer, 384 hidden dim, dynamic INT8)

## Latency Profiles (ms)
| Sequence Length (Tokens) | CPU Threads | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) |
| :--- | :--- | :--- | :--- | :--- |
| 32 | 1 | 3.04 ms | 4.06 ms | 6.46 ms |
| 32 | 2 | 3.04 ms | 3.45 ms | 3.53 ms |
| 32 | 4 | 3.05 ms | 3.20 ms | 3.28 ms |
| 64 | 1 | 4.03 ms | 4.26 ms | 4.30 ms |
| 64 | 2 | 3.28 ms | 3.68 ms | 4.27 ms |
| 64 | 4 | 3.24 ms | 3.42 ms | 3.50 ms |
| 128 | 1 | 5.38 ms | 5.66 ms | 5.70 ms |
| 128 | 2 | 4.60 ms | 4.76 ms | 4.94 ms |
| 128 | 4 | 3.99 ms | 4.08 ms | 4.18 ms |
| 256 | 1 | 8.09 ms | 9.43 ms | 9.94 ms |
| 256 | 2 | 5.97 ms | 6.66 ms | 7.05 ms |
| 256 | 4 | 5.14 ms | 5.74 ms | 6.01 ms |
