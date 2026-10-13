# ANJANEY — On-Device Inference Benchmark (Simulated)

> ⚠️ **SIMULATION NOTICE**: Measurements captured in a local environment simulating mobile CPU constraints.

- **Peak Host Memory (RSS):** 200.9 MB
- **Model Architecture:** `all-MiniLM-L6-v2` (6-layer, 384 hidden dim, dynamic INT8)

## Latency Profiles (ms)
| Sequence Length (Tokens) | CPU Threads | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) |
| :--- | :--- | :--- | :--- | :--- |
| 32 | 1 | 3.14 ms | 4.22 ms | 6.52 ms |
| 32 | 2 | 3.14 ms | 3.69 ms | 5.03 ms |
| 32 | 4 | 3.15 ms | 3.49 ms | 5.10 ms |
| 64 | 1 | 4.00 ms | 4.38 ms | 4.57 ms |
| 64 | 2 | 3.38 ms | 4.82 ms | 5.25 ms |
| 64 | 4 | 4.10 ms | 5.51 ms | 5.60 ms |
| 128 | 1 | 6.03 ms | 8.40 ms | 9.39 ms |
| 128 | 2 | 5.05 ms | 6.69 ms | 7.08 ms |
| 128 | 4 | 4.06 ms | 4.86 ms | 9.18 ms |
| 256 | 1 | 8.54 ms | 9.58 ms | 9.71 ms |
| 256 | 2 | 6.49 ms | 7.26 ms | 7.56 ms |
| 256 | 4 | 5.44 ms | 7.43 ms | 8.22 ms |
