# JEPA-KG

**Knowledge Graph Reasoning with Joint-Embedding Predictive Architecture (JEPA)**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

JEPA-KG applies the Joint-Embedding Predictive Architecture to Knowledge Graph (KG) reasoning. It fine-tunes a causal LLM with a dual-objective loss, standard language modelling combined with a JEPA alignment term, so that the model learns to predict structured KG outcomes from partial context, rather than only predicting the next token.

The primary benchmark is the **Hetionet** biomedical knowledge graph, where the task is: given a compound, predict its associated genes, diseases, side effects, pharmacological classes, and other entity relationships ranked by priority.

> **Current approach & future direction:** Currently, knowledge graphs are serialised into textual triple sequences to fit the text-based JEPA definition and enable fine-tuning of causal LLMs. In future work, we aim to move beyond this text-conversion step and directly enhance the latent space of the knowledge graphs themselves, operating on graph-native representations rather than their textual proxies.

---

## Repository Structure

```
JEPA-KG/
├── stp.py                     # Core training script — RepresentationTrainer + data loading
├── eval_hetionet.py           # Evaluation script — entity-level Precision/Recall/F1 on Hetionet
├── run_hetionet_stp.sh        # Shell script to launch Hetionet JEPA fine-tuning via torchrun
├── pyproject.toml             # Project dependencies
├── datasets/
│   ├── hetionet_train.jsonl   # Training split (compound → KG entity predictions)
│   └── hetionet_test.jsonl    # Test split
├── eval/
│   ├── eval_hetionet_jepa.jsonl          # Per-example results for JEPA fine-tuned model
│   ├── eval_hetionet_jepa_summary.json   # Aggregate metrics for JEPA model
│   ├── eval_hetionet_base.jsonl          # Per-example results for base LLaMA-3.2-1B
│   └── eval_hetionet_base_summary.json   # Aggregate metrics for base model
└── slurm/
    ├── run_hetionet_jepa.sbatch   # SLURM job for JEPA fine-tuning (4× GPU)
    └── eval_hetionet.sbatch       # SLURM job for evaluation
```

---

## How It Works

### Training Objective

The model is trained with two simultaneous objectives:

```
Total Loss = CrossEntropy(token prediction)
           + λ × (1 − cosine_similarity(z_context, z_target))
```

- **Language modelling loss**: predicts the assistant response tokens from the full conversation.
- **JEPA alignment loss** (weighted by `λ`, default `0.05`): forces the hidden representation at the end of the user/context block to align with the representation at the end of the target/answer block. This encourages the model to encode the *semantics of the answer* into its context representation before generating.

The JEPA alignment is implemented in the `RepresentationTrainer` class in `stp.py`. The `linear` mode (`--linear=random_span`) uses a random span of the full context-answer sequence as the JEPA target.

### Predictor Tokens

Learnable special tokens `<|predictor_1|>` … `<|predictor_N|>` are appended to user messages (controlled by `--predictors`). In the Hetionet experiments, 128 predictor tokens are used. These tokens act as a latent bottleneck through which the model routes its predictions.

### Data Format

Each training and test example is a three-turn conversation in JSONL format:

```json
{
  "messages": [
    {"role": "system",    "content": "...task instructions..."},
    {"role": "user",      "content": "...compound context + predictor tokens..."},
    {"role": "assistant", "content": "...ranked entity list by priority..."}
  ]
}
```

The assistant response lists predicted entities as:

```
[Priority 1] Gene Bindings (Gene)
- TP53 (Gene)
- BRCA1 (Gene)

[Priority 2] Similar Compounds (Compound)
- Aspirin (Compound)
...
```

---

## Installation

```bash
git clone <repo-url>
cd JEPA-KG
pip install -e .
```

Or install dependencies directly:

```bash
pip install torch transformers peft accelerate datasets safetensors rich pandas tqdm
```

> **Optional:** `pip install rdflib` enables RDF/Turtle/N-Triples ingestion in `JEPA-KG.py`.

---

## Hetionet Fine-Tuning

### Quick Start (local, 4 GPUs)

```bash
bash run_hetionet_stp.sh
# or with custom parameters:
bash run_hetionet_stp.sh meta-llama/Llama-3.2-1B-Instruct 2e-5 0.05 42
```

Arguments (all optional, positional):

| Position | Argument | Default |
|----------|----------|---------|
| 1 | `model_name` | `meta-llama/Llama-3.2-1B-Instruct` |
| 2 | `learning_rate` | `2e-5` |
| 3 | `lbd` (JEPA weight λ) | `0.05` |
| 4 | `seed` | `42` |

The script calls `torchrun --nproc_per_node=4 stp.py` with the following key settings:

- `--predictors=128` — 128 predictor tokens appended to user messages
- `--max_length=2048` — KG examples are longer than typical NLP tasks
- `--num_epochs=3`
- `--linear=random_span` — random-span JEPA alignment mode
- Output saved to `trained_models/ft-hetionet-jepa-<lr>-<lbd>-<seed>/`

### SLURM (HPC cluster)

```bash
sbatch slurm/run_hetionet_jepa.sbatch
```

Requests 4 GPUs on the `dc-hwai` partition. Edit the `--account` field before submitting.

---

## `stp.py` — Training Script

`stp.py` is the core training module. It implements:

| Component | Description |
|-----------|-------------|
| `load_and_prepare_dataset` | Loads JSONL, applies chat template, tokenizes, creates masked labels and user/assistant start-end indices |
| `LinearPredictor` | Optional linear projection layer (d → d) used as a latent predictor head |
| `setup_model_and_tokenizer` | Loads model + tokenizer, adds special tokens, applies LoRA, optionally initializes `LinearPredictor` |
| `RepresentationTrainer` | HuggingFace `Trainer` subclass implementing the dual-objective loss and all JEPA representation alignment modes |

#### Key CLI Flags for `stp.py`

| Flag | Description | Default |
|------|-------------|---------|
| `--model_name` | Base model (HF hub or local path) | required |
| `--train_file` | Training JSONL path | required |
| `--eval_file` | Evaluation JSONL path | — |
| `--output_dir` | Where to save the fine-tuned model | required |
| `--learning_rate` | AdamW learning rate | `2e-5` |
| `--num_epochs` | Training epochs | `3` |
| `--lbd` | JEPA loss weight (λ) | `0.05` |
| `--predictors` | Number of predictor tokens appended to user context | `0` |
| `--max_length` | Max token sequence length | `2048` |
| `--lora_rank` | LoRA rank | `16` |
| `--linear` | JEPA alignment mode: `random_span` or unset | `None` |
| `--last_token` | Token offset for representation extraction (model-specific) | `-2` |
| `--finetune_seed` | Random seed | `42` |
| `--debug` | Debug verbosity level (0 = off) | `0` |

---

## Evaluation

### Run Evaluation

```bash
# Evaluate JEPA fine-tuned model
python eval_hetionet.py \
  --model_name trained_models/ft-hetionet-jepa-2e-5-0.05-42 \
  --original_model_name meta-llama/Llama-3.2-1B-Instruct \
  --input_file datasets/hetionet_test.jsonl \
  --output_file eval/eval_hetionet_jepa.jsonl \
  --max_new_tokens 512

# Evaluate base model (zero-shot baseline)
python eval_hetionet.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --original_model_name meta-llama/Llama-3.2-1B-Instruct \
  --input_file datasets/hetionet_test.jsonl \
  --output_file eval/eval_hetionet_base.jsonl \
  --max_new_tokens 512
```

Or via SLURM:

```bash
sbatch slurm/eval_hetionet.sbatch
```

### Metrics

`eval_hetionet.py` scores entity-level **Precision**, **Recall**, and **F1** by extracting `- Name (Kind)` lines from generated and ground-truth responses and comparing them as sets. Exact-match scoring is unsuitable here because ground-truth answers are long, ranked, structured lists.

A per-priority breakdown is also computed (e.g. Gene Bindings, Side Effects, Similar Compounds, etc.).

### Results on Hetionet (199 test examples)

| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| LLaMA-3.2-1B-Instruct (base, zero-shot) | 0.009 | 0.003 | 0.005 |
| LLaMA-3.2-1B-Instruct + JEPA fine-tuning | 0.085 | 0.094 | 0.086 |

**Per-priority recall (JEPA fine-tuned):**

| Priority Section | Recall | F1 |
|-----------------|--------|----|
| Similar Diseases | 0.305 | 0.186 |
| Gene Bindings | 0.122 | 0.105 |
| Similar Compounds | 0.108 | 0.086 |
| Pharmacologic Classes | 0.071 | 0.077 |
| Side Effects | 0.017 | 0.017 |

The JEPA fine-tuned model improves recall by ~28× over the zero-shot base model, demonstrating that the representational alignment objective teaches the model the structure of KG neighbourhood prediction.

## Supported Base Models

Any HuggingFace causal LLM with a chat template can be used. The `--last_token` offset is model-specific, and controls which hidden state position is used for JEPA alignment:

| Model family | `--last_token` |
|--------------|---------------|
| `meta-llama/Llama-3.2-*` | `-2` |
| `google/gemma-*` | `-2` |
| `Qwen/Qwen*` | `-3` |
| `allenai/OLMo-2*` | `-1` |
| `deepseek-ai/DeepSeek*` | `-1` |
| `apple/OpenELM*` | `-4` |

---

## Dependencies

Managed via `pyproject.toml`. Core requirements:

```
torch >= 2.10
transformers >= 5.3
peft >= 0.18
accelerate >= 1.13
datasets >= 4.7
safetensors >= 0.7
rich >= 14.3
pandas >= 3.0
tqdm >= 4.67
```

Install all:

```bash
pip install -e .
```

---

## Conceptual Foundations

Built on ideas from:

- **Yann LeCun's Joint-Embedding Predictive Architecture (JEPA)** : the core self-supervised objective of aligning context and target representations in latent space, rather than predicting raw outputs
- **Neuro-symbolic AI and knowledge graph engineering** : combining symbolic graph structure with neural representation learning
- **Self-supervised world models and representation learning** : learning predictive models of structured environments without explicit supervision

---

## Future Directions

- **Graph-native latent representations** : move beyond serialising KGs to text; apply JEPA alignment directly in the embedding space of graph neural networks
- **Temporal KG modelling** : extend to sequences of KG snapshots, enabling the model to learn how graph structure evolves over time
- **Causal intervention simulation** : support do-calculus-style reasoning over KG state spaces to answer counterfactual questions
- **Reinforcement learning over KG state spaces** : train agents to navigate and modify KG structure guided by JEPA-based world models
