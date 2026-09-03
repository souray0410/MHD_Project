# Experiments / 实验

Experiments are separated from the versioned framework sources. Each experiment should state its target MHD version, data assumptions, dependency environment, tested paths, and the exact boundary of any reported result.

实验与版本核心源码分离。每个实验应明确目标 MHD 版本、数据假设、依赖环境、实际执行路径和结果适用边界。

```text
experiments/
├── V1/
│   └── legacy/            # original V1 pipelines and auxiliary research code
└── V4/
    └── retfound_2d/       # staged RETFound ViT-L/16 CFP/OCT validation
```

- [`V1/legacy`](V1/legacy) preserves the original V1 research layout without rewriting its source.
- [`V4/retfound_2d`](V4/retfound_2d) documents the RETFound topology, UK Biobank adapter, forward/backward checks, and saved result records.

- [`V1/legacy`](V1/legacy) 原样保留 V1 研究代码布局；
- [`V4/retfound_2d`](V4/retfound_2d) 记录 RETFound 拓扑、UK Biobank 适配、前后向检查和已有结果。
