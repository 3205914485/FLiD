# FLiD: **F**ramework for **L**abel-L**i**mited **D**ynamic Node Classification
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository is built for the paper

**PTCL: Pseudo-Label Temporal Curriculum Learning for
Label-Limited Dynamic Graph**

FLiD is a novel framework for dynamic graph learning where only final timestamp labels are available. Designed for extensibility and fairness, it supports cutting-edge research in temporal graph analysis through:

<p align="center">
  <img src='images/overview.png'>
</p>

## 🚀 Key Features
- **Novel Architecture**
  * Dual-model EM optimization with pseudo-label enhancement
  * Support for below paradigms with modular components: 
    * CFT (Copy-Final Timestamp labels) * DLS (Dynamic Label Supervision)
    * NPL (Naive Pseudo-Labels) 
    * SEM (Standard EM) 
    * PTCL-2D (PTCL with 2 Decoders) 
    * PTCL-2D
  ![Methods](images/methods.png)
  * Support for below pseudo-labels enhancement methods:
    * Confidence Score Threshold (CST)
    * Entropy of Softmax Trajectory (EST) 
    * Temporal Curriculum Learning

  * Support for below various backbone models for Label-limited Dynamic Node Classification:
    * TGAT 
    * TGN 
    * GraphMixer
    * TCL
    * DyGFormer 

  * Support for below datasets:
    |                         | Wikipedia | Reddit  | Dsub   | CoOAG  |
    |-------------------------|-----------|---------|--------|--------|
    | **Nodes**               | 9,227     | 10,984  | 150,000| 9,559  |
    | **Edges**               | 157,474   | 672,447 | 168,154| 114,337|
    | **Duration**            | 1 month   | 1 month | 1 year | 22 years|
    | **Total classes**       | 2         | 2       | 2      | 5      |
  * Adaptive weight scheduling (`gt_weight` decay)

- **Comprehensive Support**
  ```python
  # Flexible training workflows
  for k in range(args.num_em_iters):
      e_step(...)  # Expectation step
      m_step(...)  # Maximization step
      update_pseudo_labels(...)  # Label refinement
  ```

- **Research-Ready Infrastructure**
  - Automatic metric tracking (AUC/Accuracy)
  - Early stopping with model checkpointing
  - Reproducible seed management
  - Multi-run experiment support

## 📦 Installation
```bash
conda create -n flid python=3.8
conda activate flid
pip install -r requirements.txt
```

## 🛠 Usage

-  数据预处理

    使用 `preprocess` 处理数据。

- Step 1: Warmup

    1. 配置以下参数进行 warmup 训练：
    - `model_name`：选择模型（如 `TGAT`）
    - `gpu`：指定使用的 GPU（如 `5`）
    - `dataset`：选择数据集（如 `reddit` 或 `wikipedia`）
    - `threshold`：伪标签阈值（如 `0.5`）
    - `gt_weight`：伪标签的权重（如 `0.9`）

    运行 warmup 阶段：

    ```bash
    bash warmup.sh
    ```

- Step 2: 训练

    1. 配置以下参数进行训练：
    - `method`：选择训练方法（如 `PTCL`, `SEM`, `NPL`）
    - `dataset`：选择数据集（如 `reddit`, `wikipedia`, `oag`）
    - `gt_weight`：伪标签的权重（如 `0.5`）
    - `alphas`：设置不同的超参数（如 `0.1`）
    - `gpus`：指定使用的 GPU（如 `[1]`）
    - `max_tasks_per_gpu`：每个 GPU 最大任务数（如 `1`）

    运行训练脚本：

    ```bash
    bash train.sh
    ```

- 结果

    - 训练过程中，`logs/` 目录将保存训练日志。
    - 训练过程中，`results/` 目录将保存训练结果。
    - 可以根据输出的 `AUC` 和 `ACC` 评估模型性能。

## 📊 Method Comparison

| Method       | EM Steps | Pseudo-Label Strategy     | Key Characteristics                 |
|--------------|----------|---------------------------|-------------------------------------|
| **CFT**      | None     | Copy Final Labels         | Baseline with label propagation     |
| **DLS**      | None     | Full Supervision          | With dynamic labels             |
| **NPL**      | None     | Joint Optimization with generated pseudo-labels       | Single-phase training               |
| **SEM**      | Full  | 2-stage generate pseudo-labels          | Standard EM Implementation          |
| **PTCL**     | Full     | E-step generate pseudo-labels       | Dual-phase EM + Temporal Filtering  |
| **PTCL-2D**  | Full     | Dual-Decoder Architecture | Prevents confirmation bias          |


## 📂 Repository Structure
```
FLiD/
├── models/                    # Various Backbone
├── logs/                      # Training metrics & logs
│   └── {method}/              # Per-method organization
├── results/                   # Trainin results
│   └── {method}/              # Per-method organization
├── saved_models/              # Model checkpoints
├── process_data/              # Preprocess tools
├── processed_data/            # Preprocessed datasets
│
├── PTCL/                      # Core EM implementation
│   ├── EM_init.py             # Model initialization
│   ├── EM_warmup.py           # Model warmup
│   ├── E_step.py              # Expectation phase
│   └── M_step.py              # Maximization phase
│   ├── trainer.py             # Trainier initialization
│   ├── utils.py               # Important utils tools
│
├── NPL/                       # Core NPL implementation
│   ├── NPL_init.py            # Model initialization
│   ├── NPL.py                 # Training phase
│
├── SEM/                       # Core NPL implementation
│   ├── E_step.py              # Expectation phase
│   ├── M_step.py              # Maximization phase
│
├── utils/                     # Infrastructure
│   ├── DataLoader.py          # Dataset processing
│   ├── EarlyStopping.py       # Training control
│   └── load_configs.py        # Logging setup
│   └── utils.py               # Useful tools
│   └── metrics.py             # Metrics calculating tools
│
└── train.py                   # Experiment entry point
```

## 📈 Evaluation Metrics
Framework tracks multiple metrics through `log_and_save_metrics()`:
```python
# Sample metric output
2023-11-15 14:30:00 - Estep - INFO - Test Metrics:
{
    "AUC": 0.892,
    "Accuracy": 0.814,
    "Loss": 0.423
}
```

## 🤝 Contributing
We welcome contributions! Please follow our [contribution guidelines](CONTRIBUTING.md) and:
- Use consistent logging practices
- Maintain backward compatibility
- Add unit tests for new features

## 📜 Citation
If using FLiD in your research, please cite:
```bibtex
@article{flid2023,
  title={PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph},
  author={Shengtao Zhang, Haokai Zhang, Shiqi Lou, Zicheng Wang, Zinan Zeng, Yilin Wang, Minnan Luo},
  journal={Waiting},
  year={2025}
}
``` 

## License
This project is licensed under the [MIT License](LICENSE).