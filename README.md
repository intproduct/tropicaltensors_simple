# Tropi Algebra Toolkit（private）

A minimal and extensible Python toolkit for **tropical algebra** 

This package currently includes two main modules:

- **tropinum.py** — Defines `TropiNum`, tropical scalars, and basic tensor-like wrappers.
- **tropifunc.py** — Implements higher-level tropical algebra functions such as tropical matrix multiplication, reductions, contractions, and helper utilities.

The toolkit is designed to integrate smoothly with CPU/GPU backends (NumPy, MLX, torch)


## Features
- Tropical addition (max)
- Tropical multiplication (sum)
- Broadcasting-friendly tensor wrappers
- Tropical matmul implementation
- Backend-agnostic design


**Tropi Algebra Toolkit** 是一个简洁、可扩展的 **热带代数（tropical algebra）工具库**，适用于以下用途：


* **tropinum.py** — 定义 `TropiNum`、热带标量以及基础张量包装。
* **tropifunc.py** — 提供高层的热带代数函数，包括热带矩阵乘法、规约、收缩以及辅助工具函数。

本工具库设计为可轻松与 CPU/GPU 后端结合（NumPy、MLX，torch）

---

## 功能特点

* 热带加法（max）
* 热带乘法（sum）
* 支持广播的张量封装
* 热带矩阵乘法实现
* 后端无关设计（可切换 NumPy/MLX 等）

---

## 安装方法

```bash
git clone <your-repo-url>
cd <repo>
pip install -e .
```


---

## 后续计划

* Tropical einsum 实现
* Tropical tensordot
* MLX 自动求导支持
* Tropical 张量网络层
* MoE 路由结构集成
