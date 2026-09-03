# V1 Legacy Research Code / V1 历史研究代码

This directory preserves the original V1 datasets, training pipelines, comparison models, result utilities, and helper scripts from the earlier GitHub repository. Python source content was not rewritten during the repository reorganization.

本目录保存早期 GitHub 仓库中的 V1 数据处理、训练 pipeline、对照模型、结果工具和辅助脚本。仓库重排没有改写这些 Python 文件的内容。

The original layout and imports are retained for historical traceability. These files may depend on project-specific data paths and older third-party environments; they are not presented as a current turnkey example.

为保留历史可追溯性，原目录关系和 import 方式继续保留。这些脚本可能依赖项目数据路径和旧版第三方环境，不代表当前可以直接运行的新手示例。

The two main V1 modules are also exposed under stable repository names:

- [`../../../V1/MHD_Framework_V1.py`](../../../V1/MHD_Framework_V1.py), copied from `node_toolkit/node_net.py`;
- [`../../../V1/MHD_Utils_V1.py`](../../../V1/MHD_Utils_V1.py), copied from `node_toolkit/node_utils.py`.
