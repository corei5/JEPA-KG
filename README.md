# JEPA-KG: Enterprise World Model
### Neuro-Symbolic Predictive Intelligence for the Self-Understanding Enterprise

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

---

## What Is This?

JEPA-KG is a **neuro-symbolic enterprise world model**, a compact, predictive model of how an organisation's objects, rules, processes, people, systems, and external constraints interact and evolve over time.

It moves beyond standard LLMs (which predict tokens) by learning **causal enterprise dynamics**: given a structured Knowledge Graph representation of an organisational state, it predicts how that state transitions — across supply chains, compliance obligations, business processes, data pipelines, and strategic scenarios.

---

## Five Core Capabilities

| # | Task | What It Does |
|---|------|-------------|
| 1 | **Supply Chain Cascade Prediction** | Given a component delay, predicts every downstream production, financial, and contractual impact step by step |
| 2 | **Compliance Obligation Inference** | Given a product, its markets, and known regulations, infers *all* missing compliance obligations, including those not yet stated |
| 3 | **Process Violation Prediction** | Scans a running business process against its constraints and predicts which steps will breach SLAs, regulations, or policies before they do |
| 4 | **Data Quality Defect Prediction** | Analyses a data pipeline and its recent changes to predict data quality failures before they corrupt downstream reports and decisions |
| 5 | **Business Impact Simulation** | Simulates the full financial, operational, regulatory, and ESG impact of changing a supplier, material, policy, or workflow |

---

## Architecture

### Dual-Objective Training

The model optimises two losses simultaneously:

```
Total Loss = λ_lm  × CrossEntropy(token prediction)
           + λ_jepa × (1 − cosine_similarity(z_context, z_target))
```

**Generative branch (λ_lm):** Standard transformer language modelling for natural language interaction, knowledge retrieval, and prediction output generation.

**JEPA branch (λ_jepa):** Forces the model's latent representation of an observed enterprise state (z_context) to align with the latent representation of the resulting outcome (z_target). This is what makes the model learn causal enterprise dynamics rather than surface-level token co-occurrence.

### Knowledge Graph Serialisation

Enterprise state is expressed as RDF-style triples wrapped in modality tokens:

```
<|kg_start|>
(Supplier_Taiwan_Semi, delayedBy, 6_weeks)
(Microcontroller_MCU_X7, stockLevel, 3_days_supply)
(ECU_Module_B, noSubstitute, true)
(Plant_Stuttgart, dailyOutput, 850_units)
<|kg_end|>
<|observation|> Taiwan Semiconductor has notified a 6-week delay ...
[constraints: JIT policy: max 5-day buffer; OEM penalty €120k/day]
<|predictor_1|> <|predictor_2|> <|predictor_3|> <|predictor_4|>
```

### Predictor Tokens

Learnable latent transition operators inserted between context and prediction:

```
<|predictor_1|> ... <|predictor_8|>
```

During training, these tokens learn to encode enterprise state transitions in latent space. In inference, they act as a "simulation buffer" between the observed state and the predicted outcome.

### Modality Tokens

Separate namespaces for different cognitive modes:

| Token | Purpose |
|-------|---------|
| `<|kg_start|>` / `<|kg_end|>` | Knowledge Graph triple block |
| `<|observation|>` | Natural language event description |
| `<|supply_chain|>` | Supply chain domain context |
| `<|compliance|>` | Regulatory/compliance domain context |
| `<|process|>` | Business process domain context |
| `<|data_quality|>` | Data pipeline domain context |
| `<|simulation|>` | Change simulation domain context |
| `<|impact|>` | Predicted enterprise outcome |
| `<|governed_action|>` | Policy/SHACL-constrained output |

---

## Requirements

```
NVIDIA GPU (16GB VRAM minimum, 24GB+ recommended)
CUDA 11.8+
Python 3.9+
```

For CPU/MPS (slower, for development):

```
Any modern multi-core CPU or Apple Silicon Mac
```

---

## Installation

```bash
git clone https://github.com/your-org/jepa-kg.git
cd jepa-kg

pip install torch transformers peft accelerate bitsandbytes datasets rich
```

---

## Quick Start

### Run the Demo (Zero-Shot Mode)

Runs all five enterprise tasks using the base LLM with structured KG prompts, no fine-tuning required:

```bash
python enterprise_world_model.py
```

This will:

1. Load the base model (Gemma-2-2B-IT by default)
2. Display structured predictions in the terminal
3. Export scenario KG structures to `enterprise_scenarios.json`

### Enable Fine-Tuning

In `enterprise_world_model.py`, set:

```python
TRAIN_MODE = True
```

This runs LoRA fine-tuning on the five scenario templates before inference. Replace the `ground_truths` dict inside `build_training_dataset()` with real historical enterprise outcomes for production use.

### Swap the Base Model

```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"   # or any HF causal LLM
```

---



---

## Extending the Model

### Adding a New Enterprise Scenario

```python
from enterprise_world_model import EnterpriseContext, KGTriple, TaskType, SCENARIOS

SCENARIOS[TaskType.SUPPLY_CHAIN_IMPACT] = EnterpriseContext(
    task_type=TaskType.SUPPLY_CHAIN_IMPACT,
    triples=[
        KGTriple("Supplier_A", "delivers", "Component_X"),
        KGTriple("Component_X", "delayedBy", "3_weeks"),
        # add all relevant triples
    ],
    observation="Your natural language description of the event.",
    constraints=[
        "Policy or regulatory constraint 1",
        "SLA constraint 2",
    ],
    metadata={"domain": "your_domain"},
)
```

### Connecting to a Real Knowledge Graph

Replace the static `SCENARIOS` dict with a live KG query:

```python
# SPARQL query to eccenca Corporate Memory (or any SPARQL endpoint)
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://your-kg-endpoint/sparql")
sparql.setQuery("""
    SELECT ?s ?p ?o WHERE {
        ?s ?p ?o .
        FILTER(?s = <urn:supplier:taiwan-semi>)
    }
""")
results = sparql.query().convert()
triples = [KGTriple(r["s"]["value"], r["p"]["value"], r["o"]["value"])
           for r in results["results"]["bindings"]]
```

### Training on Real Historical Data

```python
# In build_training_dataset(), replace ground_truths with real outcomes:
ground_truths[TaskType.SUPPLY_CHAIN_IMPACT] = load_from_corporate_memory(
    query="SELECT outcome WHERE disruption_id = 'event_2024_q3_taiwan'"
)
```

---

## Evaluation

| Domain | Predictive Task | Key Metric |
|--------|----------------|------------|
| Automotive | Supply Chain Cascade | Entity recall vs. actual affected systems |
| Life Sciences | Compliance Inference | Obligation recall vs. legal review checklist |
| Financial Services | Process Violation | Precision/recall of constraint breach prediction |
| Data Engineering | Data Quality Defect | Defect detection rate before pipeline run |
| Pharmaceuticals | Business Simulation | Decision alignment vs. expert committee outcome |

---

## Research Background

The long-term goal is to evolve enterprise knowledge platforms from passive data fabrics into active self-simulating world models capable of causal reasoning, predictive simulation, policy-constrained planning, and autonomous enterprise agency.

---

## Conceptual Foundations

Built on ideas from:

- Yann LeCun's Joint-Embedding Predictive Architecture (JEPA)
- Neuro-symbolic AI and knowledge graph engineering
- Self-supervised world models and representation learning
- RDF/OWL/SHACL enterprise semantic standards
- Constraint-guided and governance-aware AI systems

---

## Future Directions

- Temporal enterprise state modeling with graph sequence encoders
- Multi-agent organisational simulation
- Reinforcement learning over KG state spaces
- SHACL-guided constrained decoding
- Live KG ingestion from Corporate Memory / triplestore endpoints
- Causal intervention simulation (do-calculus over enterprise KGs)
- Autonomous governance-aware copilots

---

## License

MIT License. See `LICENSE` for details.

---

## Vision

> "The future enterprise will not merely store knowledge, it will simulate itself."

JEPA-KG is a step toward organisations that understand their own internal dynamics: systems that can look at a disruption, a regulatory change, or a strategic decision and reason through the consequences, before they happen.
