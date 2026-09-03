# Tests / 测试

The current test suite primarily targets V4 behavior, V3-to-V4 migration, model equivalence, lifecycle handling, and distributed smoke paths.

当前测试主要覆盖 V4 行为、V3→V4 迁移、模型等价性、生命周期和分布式 smoke 路径。

```bash
pytest -q tests
```

Distributed and GPU smoke scripts require the corresponding PyTorch distributed environment. A test count or pass claim should only be published after running the exact current source snapshot; historical README claims are not a substitute for a fresh run.

分布式与 GPU smoke 脚本需要相应 PyTorch 环境。只有对当前精确源码重新运行后，才能发布通过数量；历史 README 中的记录不能替代当前测试。

Repository reorganization changes paths and documentation only. Existing test source is preserved unchanged in this maintenance pass.

本次仓库整理只调整路径和文档，现有测试源码保持不变。
