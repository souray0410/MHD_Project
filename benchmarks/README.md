# Benchmarks / 性能基准

Benchmarks measure framework overhead against equivalent PyTorch computation. They are evidence for a specific model, device, software environment, and sampling method—not universal performance guarantees.

Benchmark 用于比较 MHD 调度与等价 PyTorch 计算的开销。结果只适用于明确记录的模型、设备、软件环境和采样方法，不应推广为普遍性能保证。

- [`benchmark_v4_overhead.py`](benchmark_v4_overhead.py): V4 Transformer-oriented forward and training overhead measurement.

Run from the repository root in a compatible PyTorch environment and record the hardware and software versions together with any result.

请在仓库根目录和兼容的 PyTorch 环境中运行，并在引用结果时同时记录硬件和软件版本。
