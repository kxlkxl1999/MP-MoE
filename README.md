# MP-MoE
This is the code file for the paper: Breaking the Echo Chamber: A Dynamic Ensemble Pruning Perspective on MoE. The code in the paper is trained using eight Ascend 910B NPUs. For detailed configuration information, please refer to https://gitee.com/ascend/MindSpeed-LLM. This project code is a demo code for an NVIDIA environment.

## Repository Structure

```
mp-moe/
├── megatron-lm/                        # Megatron-LM modifications (6 files)
├── lm-evaluation-harness/              # Evaluation framework modifications (3 files)
├── flame-moe/                          # Configs and scripts
│   ├── configs/                        # Model and training configs
│   └── scripts/                        # Training, evaluation, ablation scripts
└── patches/                            # Patch files for easy application
```

## Base Repositories

- **Megatron-LM**: `https://github.com/yuzc19/Megatron-LM` (branch: `multi-nodes`, commit: `cbaf684`)
- **lm-evaluation-harness**: `https://github.com/yuzc19/lm-evaluation-harness` (branch: `megatron`, commit: `0c8c0d8`)

## Installation

### Method 1: Apply Patches (Recommended)

```bash
git clone -b multi-nodes https://github.com/yuzc19/Megatron-LM.git
git clone -b megatron https://github.com/yuzc19/lm-evaluation-harness.git

cd Megatron-LM && git apply ../mp-moe/patches/megatron-lm.patch
cd ../lm-evaluation-harness && git apply ../mp-moe/patches/lm-evaluation-harness.patch
```

### Method 2: Manual Copy

Copy files from `megatron-lm/` and `lm-evaluation-harness/` to corresponding locations in base repositories.

## Modifications Overview

### Megatron-LM (6 files)

| File                    | Description                                                                                   |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| `router.py`             | MoE router with MoPE logic, static/dynamic expert mask (co-occurrence/random), OTEP selection |
| `moe_layer.py`          | Core MoE layer with routing statistics aggregation and disk saving                            |
| `OT_pruning.py`         | **NEW** - Optimal Transport based expert pruning (batched version)                            |
| `transformer_config.py` | Extended config for statistics/plotting (e.g., `moe_plot_every`)                              |
| `arguments.py`          | Extended CLI arguments for MoPE                                                               |
| `moe_module_specs.py`   | `MoEOrMoPE` selector based on `moe_router_load_balancing_type == otep`                        |

### lm-evaluation-harness (3 files)

| File                     | Description                                             |
| ------------------------ | ------------------------------------------------------- |
| `megatron_lm.py`         | MegatronLM adapter with `generate_until` implementation |
| `_fewshot_template_yaml` | BBH fewshot configuration                               |
| `gsm8k.yaml`             | GSM8K task configuration                                |

### Scripts (flame-moe/)

- **Training**: `scripts/release/*_local.sh` - Local torchrun training (auto-detect GPU, adjust EP)
- **Dataset**: `scripts/dataset/*fineweb*` - FineWeb-Edu download/tokenization pipeline
- **Evaluation**: `scripts/eval_benchmark.sh`, `scripts/run_all_evals.sh`
- **Ablation**: `scripts/ablation/*.py` - Expert similarity, t-SNE/PCA/CKA visualization

## Usage

```bash
# Training
bash scripts/release/flame-moe-38m_local.sh

# Evaluation
bash scripts/eval_benchmark.sh

# Ablation
python scripts/ablation/plot_similarity_tsne.py
```

## License

- **Megatron-LM**: BSD 3-Clause License, Copyright (c) NVIDIA CORPORATION
- **lm-evaluation-harness**: MIT License, Copyright (c) EleutherAI

See [LICENSE](LICENSE) for full texts.
