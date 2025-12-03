This project is forked from https://github.com/verlab/accelerated_features

# README（适用于 Windows + uv 环境）
## 如何在 Windows 上复现环境
### 1. 安装 uv（如果尚未安装）
官方Git安装，也可以在
https://github.com/astral-sh/uv 下载 Release 安装包
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
验证：
```
uv --version
```
### 2. 克隆仓库
```
git clone git@github.com:Yiming280/Xfeat_accelerated_features.git
cd Xfeat_accelerated_features
```

### 3. 根据 uv.lock 完整复现环境

uv 会根据 .python-version 中的版本自动选择 Python（本项目通常为 Python 3.8）。

执行：
```
uv sync
```

uv 会自动：

- 安装指定 Python 版本（如果你启用了 uv 的 Python 下载功能）

- 创建 .venv 虚拟环境

- 根据 uv.lock 安装所有依赖（精准版本）

- 确保环境与原作者完全一致

### 4. 激活虚拟环境
```
.\.venv\Scripts\activate
```