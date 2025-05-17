# FLiD: **F**ramework for **L**abel-L**i**mited **D**ynamic Node Classification
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository is built for the paper

PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph

FLiD is a novel framework for dynamic graph learning where only final timestamp labels are available. Designed for extensibility and fairness, it supports cutting-edge research in temporal graph analysis through:

<p align="center">
  <img src='images/overview.png'>
</p>

## 🚀 Key Features
- **Novel Architecture**
  * Dual-model EM optimization with pseudo-label enhancement
  * Support for below paradigms with modular components: 
    * CFT (Copy-Final Timestamp labels) 
    * DLS (Dynamic Label Supervision)
    * NPL (Naive Pseudo-Labels) 
    * SEM (Standard EM) 
    * PTCL-2D (PTCL with 2 Decoders) 
    * PTCL
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

## 🛠 Usage

- **Requirements**
  To install requirements:
    ```bash
    conda create --name <env> --file requirements.txt
    ```
- **Data Preprocessing**

    Use the `preprocess.ipynb` script to process the data. Or just skip the step if the dataset has been saved in `processed_data`. We provide `wikipedia.zip` here.
    
- **Training**
  - **Step 1: Warmup**

    1. Configure the following parameters for the warmup training:
    - `method`: Choose the method (e.g., `ptcl`) 
    - `mmodel_name`: Choose the model-backbone (e.g., `TGAT`)
    - `gpu`: Specify the GPU to use (e.g., `0`)
    - `dataset_name`: Choose the dataset (e.g., `reddit` or `wikipedia`) 
    - `warmup_m_train`: warmup-train-backbone (e.g., `1`)
    - `warmup_e_train`: warmup-train-decoder (e.g., `1`)
    - `num_epochs_e_warmup`: warmup epochs (e.g., `100`)
    - `num_epochs_m_warmup`: warmup epochs (e.g., `100`)
    1. Run the warmup step :

    ```bash
    python train.py
    ```

  - **Step 2: Train**

    1. Configure the following parameters for training:
    - `method`: Choose the training method (e.g., `ptcl`, `sem`, `npl`)
    - `dataset_name`: Choose the dataset (e.g., `reddit`, `wikipedia`, `oag`)
    - `gt_weight`: gourd-truth label weight ($\beta$ in the paper) (e.g., `0.5`)
    - `alphas`: Exp decay for Temporal Curriculum learning ($\gamma$ in the paper)(e.g., `0.1`)
    - `gpus`: Specify the GPUs to use (e.g., `0`)
    - `warmup_m_train`: warmup-train-backbone (e.g., `0`)
    - `warmup_e_train`: warmup-train-decoder (e.g., `0`)
    - `num_epochs_e_step`: number of epochs of E step (e.g., `200`)
    - `num_epochs_m_step`: number of epochs of M step (e.g., `100`)
    - `num_em_iters`: number of iters of E-M step (e.g., `30`)
    - `ps_filter`: pseudo-labels filter mechanism (e.g., `entropy` for EST,  `None` for TCL)
    2. Run the training script:

    ```bash
    python train.py
    ```
- **Evaluation**
  
  follow the train-configure:
    ```bash
    python eval.py
    ```
- **Results**

    - During training, the `logs/` directory will save the training logs.
    - The `results/` directory will store the training results.
    - You can evaluate the model performance using the output `AUC` and `ACC`.

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
├── train.py                   # Experiment train python
└── eval.py                    # Experiment evaluate python
```

## 📈 Evaluation Metrics
Framework tracks multiple metrics through `log_and_save_metrics()`:
```python
# Sample metric output
2025-1-15 14:30:00 - Estep - INFO - Test Metrics:
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
@article{ptcl2025,
  title={PTCL: Pseudo-Label Temporal Curriculum Learning for Label-Limited Dynamic Graph},
  author={Shengtao Zhang, Haokai Zhang, Shiqi Lou, Zicheng Wang, Zinan Zeng, Yilin Wang, Minnan Luo},
  year={2025}
}
``` 

## License
This project is licensed under the [MIT License](LICENSE).