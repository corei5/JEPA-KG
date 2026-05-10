"""
Enterprise World Model - JEPA-KG
=================================
A neuro-symbolic predictive system for enterprise intelligence.

Capabilities:
  1. Predict downstream supply-chain effects of a delayed component
  2. Infer missing compliance obligations from product/geography/regulation context
  3. Predict which process step will violate a constraint
  4. Predict likely data-quality defects before they surface
  5. Simulate business impact of changing a supplier, material, policy, or workflow

Requirements:
  pip install torch transformers peft accelerate bitsandbytes datasets rich

Usage:
  python enterprise_world_model.py
"""

import json
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

console = Console()

# ---------------------------------------------------------------------------
# 1. SPECIAL TOKENS
# ---------------------------------------------------------------------------

PREDICTOR_TOKENS = [f"<|predictor_{i}|>" for i in range(1, 9)]

MODALITY_TOKENS = {
    "kg_start":        "<|kg_start|>",
    "kg_end":          "<|kg_end|>",
    "observation":     "<|observation|>",
    "governed_action": "<|governed_action|>",
    "supply_chain":    "<|supply_chain|>",
    "compliance":      "<|compliance|>",
    "process":         "<|process|>",
    "data_quality":    "<|data_quality|>",
    "simulation":      "<|simulation|>",
    "impact":          "<|impact|>",
}

ALL_SPECIAL_TOKENS = PREDICTOR_TOKENS + list(MODALITY_TOKENS.values())


# ---------------------------------------------------------------------------
# 2. TASK TYPES
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    SUPPLY_CHAIN_IMPACT     = "supply_chain_impact"
    COMPLIANCE_INFERENCE    = "compliance_inference"
    PROCESS_VIOLATION       = "process_violation"
    DATA_QUALITY_PREDICTION = "data_quality_prediction"
    BUSINESS_SIMULATION     = "business_simulation"


# ---------------------------------------------------------------------------
# 3. ENTERPRISE KNOWLEDGE GRAPH SCHEMA
# ---------------------------------------------------------------------------

@dataclass
class KGTriple:
    subject: str
    predicate: str
    obj: str

    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.obj})"


@dataclass
class EnterpriseContext:
    """Structured enterprise context fed into the world model."""
    task_type: TaskType
    triples: list[KGTriple]
    observation: str
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> str:
        kg_tok = MODALITY_TOKENS
        kg_block = " ".join(str(t) for t in self.triples)
        constraint_block = "; ".join(self.constraints) if self.constraints else "none"
        return (
            f"{kg_tok['kg_start']} {kg_block} {kg_tok['kg_end']} "
            f"{kg_tok['observation']} {self.observation} "
            f"[constraints: {constraint_block}] "
            f"{' '.join(PREDICTOR_TOKENS[:4])}"
        )


# ---------------------------------------------------------------------------
# 4. PREDEFINED ENTERPRISE SCENARIOS
# ---------------------------------------------------------------------------

SCENARIOS: dict[TaskType, EnterpriseContext] = {

    # -----------------------------------------------------------------------
    TaskType.SUPPLY_CHAIN_IMPACT: EnterpriseContext(
        task_type=TaskType.SUPPLY_CHAIN_IMPACT,
        triples=[
            KGTriple("Supplier_Taiwan_Semi", "produces",       "Microcontroller_MCU_X7"),
            KGTriple("Microcontroller_MCU_X7", "usedIn",       "ECU_Module_B"),
            KGTriple("ECU_Module_B",           "partOf",       "Vehicle_Platform_Z"),
            KGTriple("Vehicle_Platform_Z",     "builtAt",      "Plant_Stuttgart"),
            KGTriple("Supplier_Taiwan_Semi",   "delayedBy",    "6_weeks"),
            KGTriple("Microcontroller_MCU_X7", "stockLevel",   "3_days_supply"),
            KGTriple("Plant_Stuttgart",        "dailyOutput",  "850_units"),
            KGTriple("ECU_Module_B",           "noSubstitute", "true"),
        ],
        observation=(
            "Taiwan Semiconductor has notified a 6-week delay on Microcontroller MCU-X7 "
            "due to fab contamination. Current warehouse stock covers 3 production days."
        ),
        constraints=[
            "JIT inventory policy: max 5-day buffer allowed",
            "OEM contract: line stoppage penalty €120k/day",
            "REACH regulation: substitute components require re-qualification",
        ],
        metadata={"domain": "automotive", "plant": "Stuttgart", "urgency": "critical"},
    ),

    # -----------------------------------------------------------------------
    TaskType.COMPLIANCE_INFERENCE: EnterpriseContext(
        task_type=TaskType.COMPLIANCE_INFERENCE,
        triples=[
            KGTriple("Product_MedDevice_Alpha", "category",      "Class_IIb_Medical_Device"),
            KGTriple("Product_MedDevice_Alpha", "marketedIn",    "EU"),
            KGTriple("Product_MedDevice_Alpha", "marketedIn",    "USA"),
            KGTriple("Product_MedDevice_Alpha", "marketedIn",    "Japan"),
            KGTriple("Product_MedDevice_Alpha", "contains",      "Software_AI_Module"),
            KGTriple("Software_AI_Module",      "type",          "Machine_Learning_Decision"),
            KGTriple("Product_MedDevice_Alpha", "dataProcessed", "Patient_Personal_Data"),
            KGTriple("Regulation_MDR_EU",       "appliesTo",     "Class_IIb_Medical_Device"),
            KGTriple("Regulation_FDA_510k",     "appliesTo",     "Class_IIb_Medical_Device"),
        ],
        observation=(
            "MedDevice Alpha is a Class IIb device incorporating an AI decision module "
            "processing patient data. It is being launched in EU, USA, and Japan simultaneously."
        ),
        constraints=[
            "EU AI Act: high-risk AI systems require conformity assessment",
            "GDPR Article 22: automated decision-making restrictions",
            "FDA AI/ML SaMD guidance: predetermined change control plan required",
        ],
        metadata={"domain": "life_sciences", "product_type": "medical_device"},
    ),

    # -----------------------------------------------------------------------
    TaskType.PROCESS_VIOLATION: EnterpriseContext(
        task_type=TaskType.PROCESS_VIOLATION,
        triples=[
            KGTriple("Process_LoanApproval",   "step_1",        "KYC_Identity_Verification"),
            KGTriple("Process_LoanApproval",   "step_2",        "Credit_Score_Check"),
            KGTriple("Process_LoanApproval",   "step_3",        "Affordability_Assessment"),
            KGTriple("Process_LoanApproval",   "step_4",        "Risk_Scoring"),
            KGTriple("Process_LoanApproval",   "step_5",        "Approval_Decision"),
            KGTriple("KYC_Identity_Verification", "SLA",        "24h"),
            KGTriple("Credit_Score_Check",     "externalAPI",   "Experian_API"),
            KGTriple("Experian_API",           "availability",  "94_percent"),
            KGTriple("Affordability_Assessment", "requires",    "Income_Verification_Doc"),
            KGTriple("Risk_Scoring",           "model",         "ML_Model_v3_2"),
            KGTriple("ML_Model_v3_2",          "drift_status",  "high_drift_detected"),
            KGTriple("Process_LoanApproval",   "volume_today",  "4200_applications"),
        ],
        observation=(
            "Loan approval process running at 4,200 applications today. "
            "Experian API shows 94% availability this week. "
            "ML Risk Model v3.2 has flagged high feature drift. "
            "Income document uploads are failing intermittently."
        ),
        constraints=[
            "FCA CONC rule: decision must be made within 72h of application",
            "GDPR: automated credit decisions must be explainable",
            "Internal SLA: KYC must complete within 24h",
            "Model governance: drifted models must not issue binding decisions",
        ],
        metadata={"domain": "financial_services", "process": "loan_origination"},
    ),

    # -----------------------------------------------------------------------
    TaskType.DATA_QUALITY_PREDICTION: EnterpriseContext(
        task_type=TaskType.DATA_QUALITY_PREDICTION,
        triples=[
            KGTriple("Pipeline_CRM_to_DWH",    "source",        "Salesforce_CRM"),
            KGTriple("Pipeline_CRM_to_DWH",    "target",        "Snowflake_DWH"),
            KGTriple("Pipeline_CRM_to_DWH",    "schedule",      "daily_02:00_UTC"),
            KGTriple("Salesforce_CRM",         "recentChange",  "Custom_Field_Migration_v4"),
            KGTriple("Custom_Field_Migration_v4", "status",     "completed_yesterday"),
            KGTriple("Pipeline_CRM_to_DWH",    "lastRun",       "success_3_days_ago"),
            KGTriple("Snowflake_DWH",          "downstream",    "Revenue_Dashboard"),
            KGTriple("Revenue_Dashboard",      "usedBy",        "CFO_Board_Report"),
            KGTriple("CFO_Board_Report",       "scheduledAt",   "tomorrow_09:00"),
            KGTriple("Field_AccountRevenue",   "mappingStatus", "unmapped_after_migration"),
            KGTriple("Field_OpportunityStage", "valueSet",      "changed_in_migration"),
        ],
        observation=(
            "CRM pipeline runs nightly. A custom field migration was completed yesterday. "
            "The CFO Board Report is due tomorrow morning and pulls from the data warehouse. "
            "AccountRevenue field mapping is flagged as unmapped post-migration."
        ),
        constraints=[
            "Data SLA: DWH must be fresh by 06:00 UTC",
            "BCBS 239: financial data lineage must be traceable",
            "Board report: zero tolerance for stale or incorrect revenue figures",
        ],
        metadata={"domain": "data_engineering", "criticality": "board_level"},
    ),

    # -----------------------------------------------------------------------
    TaskType.BUSINESS_SIMULATION: EnterpriseContext(
        task_type=TaskType.BUSINESS_SIMULATION,
        triples=[
            KGTriple("Supplier_ChemCo_A",      "supplies",      "Solvent_X22"),
            KGTriple("Solvent_X22",            "usedIn",        "Manufacturing_Process_P3"),
            KGTriple("Solvent_X22",            "REACHStatus",   "SVHC_Candidate"),
            KGTriple("Supplier_ChemCo_A",      "contractValue", "EUR_2.4M_annual"),
            KGTriple("Supplier_GreenChem_B",   "offers",        "BioSolvent_Y11"),
            KGTriple("BioSolvent_Y11",         "REACHStatus",   "fully_compliant"),
            KGTriple("BioSolvent_Y11",         "pricePremium",  "18_percent"),
            KGTriple("BioSolvent_Y11",         "qualStatus",    "not_yet_qualified"),
            KGTriple("Manufacturing_Process_P3", "output",      "Product_Line_Pharma_API"),
            KGTriple("Product_Line_Pharma_API", "annualRevenue","EUR_28M"),
            KGTriple("EU_REACH_Restriction",   "effectiveDate", "2026_Q3"),
        ],
        observation=(
            "Current supplier ChemCo A provides Solvent X22, which is on the SVHC candidate list. "
            "EU REACH restriction effective Q3 2026 will ban its use in pharmaceutical manufacturing. "
            "Green alternative BioSolvent Y11 exists but is 18% more expensive and not yet qualified."
        ),
        constraints=[
            "REACH SVHC: authorisation required from 2026-Q3",
            "GMP guidelines: solvent change requires full process revalidation (12-18 months)",
            "Financial constraint: COGS increase >15% requires board approval",
            "ESG target: reduce SVHC chemicals by 100% by 2027",
        ],
        metadata={"domain": "pharmaceuticals", "change_type": "supplier_material_swap"},
    ),
}


# ---------------------------------------------------------------------------
# 5. SYSTEM PROMPTS PER TASK
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[TaskType, str] = {

    TaskType.SUPPLY_CHAIN_IMPACT: textwrap.dedent("""
        You are an Enterprise World Model specialising in supply chain risk intelligence.
        Given a Knowledge Graph describing a component delay, analyse ALL downstream effects.

        Your response MUST include:
        IMPACT CHAIN: Step-by-step cascade from the delay to final business impact.
        AFFECTED ENTITIES: Every product line, plant, customer order affected.
        TIME HORIZON: Impact timeline (Day 1, Week 1, Week 2-6, Post-6 weeks).
        FINANCIAL EXPOSURE: Estimated cost/revenue at risk.
        MITIGATION OPTIONS: Ranked alternatives with feasibility and lead time.
        RISK SCORE: Overall supply chain risk score (1-10) with justification.
    """).strip(),

    TaskType.COMPLIANCE_INFERENCE: textwrap.dedent("""
        You are an Enterprise World Model specialising in regulatory compliance intelligence.
        Given a Knowledge Graph of a product, its markets, and known regulations,
        INFER ALL MISSING compliance obligations that the enterprise must fulfil.

        Your response MUST include:
        IDENTIFIED OBLIGATIONS: Every regulation/standard that applies, including those
            not explicitly stated in the KG (infer from product type + geography + data type).
        GAP ANALYSIS: Obligations not yet addressed in the KG context.
        OBLIGATION PRIORITY: Critical / High / Medium / Low with rationale.
        TIMELINE: Regulatory deadlines and phased requirements.
        RECOMMENDED ACTIONS: Concrete steps to achieve compliance.
    """).strip(),

    TaskType.PROCESS_VIOLATION: textwrap.dedent("""
        You are an Enterprise World Model specialising in process risk and constraint analysis.
        Given a Knowledge Graph of a business process with its steps, SLAs, and current state,
        PREDICT which steps will violate constraints and why.

        Your response MUST include:
        VIOLATION PREDICTIONS: Which exact steps will breach which constraints.
        ROOT CAUSES: Underlying signals in the KG that indicate each violation.
        PROBABILITY: Likelihood of each violation (High/Medium/Low) with evidence.
        IMPACT: What happens if the violation occurs (regulatory, financial, operational).
        PREVENTION ACTIONS: Specific interventions to prevent each predicted violation.
    """).strip(),

    TaskType.DATA_QUALITY_PREDICTION: textwrap.dedent("""
        You are an Enterprise World Model specialising in data quality and pipeline risk.
        Given a Knowledge Graph of a data pipeline, its recent changes, and downstream consumers,
        PREDICT data quality defects BEFORE they surface in production.

        Your response MUST include:
        PREDICTED DEFECTS: Specific data quality issues likely to occur (null fields,
            broken mappings, stale data, schema mismatches, incorrect aggregations).
        AFFECTED DOWNSTREAM: Which reports, dashboards, and decisions will be corrupted.
        DETECTION SIGNALS: What indicators in the KG suggested each defect.
        SEVERITY: Impact rating per defect (Critical / High / Medium).
        REMEDIATION: Steps to fix each defect before downstream consumers are hit.
    """).strip(),

    TaskType.BUSINESS_SIMULATION: textwrap.dedent("""
        You are an Enterprise World Model specialising in strategic change simulation.
        Given a Knowledge Graph describing a proposed supplier/material/policy change,
        SIMULATE the full business impact across financial, operational, regulatory, and ESG dimensions.

        Your response MUST include:
        SCENARIO ANALYSIS: What happens if the change is made vs. not made.
        FINANCIAL IMPACT: Cost delta, revenue risk, capital requirement.
        OPERATIONAL IMPACT: Process disruption, lead times, capacity effects.
        REGULATORY IMPACT: Compliance obligations triggered or resolved.
        ESG IMPACT: Sustainability, carbon, and social effects.
        DECISION RECOMMENDATION: Go / No-Go / Phased with conditions.
        IMPLEMENTATION ROADMAP: Sequenced actions with owners and timelines.
    """).strip(),
}


# ---------------------------------------------------------------------------
# 6. MODEL INITIALISATION
# ---------------------------------------------------------------------------

def initialize_world_model(model_name: str = "google/gemma-2-2b-it"):
    """
    Load base LLM and apply LoRA adapters for parameter-efficient fine-tuning.
    Adds all enterprise special tokens to the vocabulary.
    """
    console.print(f"[bold cyan]Loading tokenizer:[/bold cyan] {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ALL_SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print(f"[bold cyan]Loading model:[/bold cyan] {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# 7. DATA COLLATOR — ENTERPRISE MULTI-TASK
# ---------------------------------------------------------------------------

class EnterpriseWorldModelDataCollator:
    """
    Converts EnterpriseContext instances into training batches.

    Each sample encodes:
      - A KG-structured context (observation + triples + constraints)
      - A task-specific target (ground-truth enterprise outcome)

    The collator tracks context_end_idx and target_end_idx token positions
    so the JEPA branch can extract the correct hidden states for alignment.
    """

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode_sample(self, context: EnterpriseContext, target: str) -> dict:
        full_text = f"{context.serialize()} {MODALITY_TOKENS['impact']} {target}"
        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Locate boundary tokens for JEPA loss
        context_text = context.serialize()
        context_enc = self.tokenizer(
            context_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        context_len = min(context_enc["input_ids"].shape[1] - 1, self.max_length - 2)
        target_len  = min(encoding["input_ids"].shape[1] - 1,   self.max_length - 2)

        return {
            "input_ids":       encoding["input_ids"].squeeze(0),
            "attention_mask":  encoding["attention_mask"].squeeze(0),
            "context_end_idx": context_len,
            "target_end_idx":  target_len,
        }

    def __call__(self, examples: list[dict]) -> dict:
        input_ids      = torch.stack([ex["input_ids"]      for ex in examples])
        attention_mask = torch.stack([ex["attention_mask"] for ex in examples])
        context_ends   = [ex["context_end_idx"] for ex in examples]
        target_ends    = [ex["target_end_idx"]  for ex in examples]
        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "labels":          input_ids.clone(),
            "context_end_idx": context_ends,
            "target_end_idx":  target_ends,
        }


# ---------------------------------------------------------------------------
# 8. JEPA-KG TRAINER
# ---------------------------------------------------------------------------

class JEPAKGTrainer(Trainer):
    """
    Dual-objective trainer:

      Loss = λ_lm  * CrossEntropy(token prediction)
           + λ_jepa * (1 - cosine_similarity(z_context, z_target))

    The JEPA branch forces the model to align its latent representation of
    the observed enterprise state with the latent representation of the
    ground-truth outcome — teaching causal enterprise dynamics, not just
    next-token statistics.
    """

    def __init__(self, *args, jepa_weight: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.jepa_weight = jepa_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels         = inputs.pop("labels")
        context_ends   = inputs.pop("context_end_idx")
        target_ends    = inputs.pop("target_end_idx")

        outputs = model(
            **inputs,
            labels=labels,
            output_hidden_states=True,
        )

        lm_loss = outputs.loss
        hidden  = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)

        # Safely clamp indices
        B = hidden.shape[0]
        jepa_losses = []
        for i in range(B):
            c_idx = min(context_ends[i] if isinstance(context_ends, list)
                        else context_ends, hidden.shape[1] - 1)
            t_idx = min(target_ends[i]  if isinstance(target_ends,  list)
                        else target_ends,  hidden.shape[1] - 1)
            z_ctx = hidden[i, c_idx, :]   # latent of enterprise context state
            z_tgt = hidden[i, t_idx, :]   # latent of enterprise outcome state
            sim   = F.cosine_similarity(z_ctx.unsqueeze(0), z_tgt.unsqueeze(0))
            jepa_losses.append(1.0 - sim)

        jepa_loss = torch.stack(jepa_losses).mean()
        total_loss = lm_loss + self.jepa_weight * jepa_loss

        return (total_loss, outputs) if return_outputs else total_loss


# ---------------------------------------------------------------------------
# 9. INFERENCE ENGINE
# ---------------------------------------------------------------------------

class EnterpriseWorldModelInference:
    """
    Run zero-shot or fine-tuned inference for any of the five enterprise tasks.
    Falls back gracefully to CPU/MPS if no CUDA GPU is available.
    """

    def __init__(self, model, tokenizer, max_new_tokens: int = 512):
        self.model          = model
        self.tokenizer      = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device         = next(model.parameters()).device

    def predict(self, context: EnterpriseContext) -> str:
        system_prompt = SYSTEM_PROMPTS[context.task_type]
        kg_input      = context.serialize()

        prompt = (
            f"<|system|>\n{system_prompt}\n\n"
            f"<|enterprise_context|>\n{kg_input}\n\n"
            f"<|world_model_prediction|>\n"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.92,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# 10. TRAINING PIPELINE
# ---------------------------------------------------------------------------

def build_training_dataset(
    tokenizer,
    max_length: int = 256,
) -> tuple[list[dict], EnterpriseWorldModelDataCollator]:
    """
    Build a minimal multi-task training dataset from the five scenario templates.
    In production, replace ground-truth targets with real enterprise outcomes
    extracted from Corporate Memory / historical KG snapshots.
    """

    # Ground-truth targets (would come from historical enterprise data in production)
    ground_truths: dict[TaskType, str] = {

        TaskType.SUPPLY_CHAIN_IMPACT: (
            "IMPACT CHAIN: MCU-X7 stockout at Day 3 → ECU Module B production halt → "
            "Vehicle Platform Z line stoppage at Plant Stuttgart → 850 units/day lost. "
            "FINANCIAL EXPOSURE: €120k/day contract penalty + €72M revenue at risk over 6 weeks. "
            "MITIGATION: (1) Airfreight partial stock from Supplier Malaysia (lead 8 days, cost +€340k); "
            "(2) Activate ECU Module B substitute design with alternative MCU (requires 4-week REACH re-qual); "
            "(3) Negotiate OEM contract force-majeure clause to suspend penalty. "
            "RISK SCORE: 9/10 — critical path component with no qualified substitute."
        ),

        TaskType.COMPLIANCE_INFERENCE: (
            "INFERRED OBLIGATIONS: "
            "(1) EU MDR Article 52 — conformity assessment by Notified Body required (Class IIb). "
            "(2) EU AI Act Article 43 — high-risk AI conformity assessment mandatory for ML decision module. "
            "(3) GDPR Article 35 — Data Protection Impact Assessment required (patient data + automated decisions). "
            "(4) FDA 510(k) clearance + AI/ML SaMD Predetermined Change Control Plan. "
            "(5) Japan PMDA — medical device approval + AI accountability documentation. "
            "(6) ISO 13485 QMS certification across all three markets. "
            "CRITICAL GAPS: AI Act conformity assessment not initiated; PDCA plan missing. "
            "TIMELINE: EU AI Act enforcement August 2026 — action required immediately."
        ),

        TaskType.PROCESS_VIOLATION: (
            "VIOLATION PREDICTIONS: "
            "(1) Step 2 Credit Score Check — HIGH probability: Experian API at 94% availability "
            "will cause ~252 timeout failures today at 4,200 volume, breaching FCA 72h SLA. "
            "(2) Step 4 Risk Scoring — CRITICAL: ML Model v3.2 high drift means decisions are unreliable; "
            "issuing binding approvals violates model governance policy and GDPR explainability. "
            "(3) Step 3 Affordability Assessment — MEDIUM: income document upload failures will create "
            "incomplete assessments, causing SLA breaches for ~180 applications. "
            "PREVENTION: Halt ML Model v3.2 decisions; route to manual review. "
            "Implement Experian fallback to Equifax API. Trigger income doc re-upload workflow."
        ),

        TaskType.DATA_QUALITY_PREDICTION: (
            "PREDICTED DEFECTS: "
            "(1) CRITICAL — AccountRevenue field NULL after migration: pipeline will load NULLs "
            "into DWH; CFO Board Report revenue figures will be zero or missing. "
            "(2) HIGH — OpportunityStage values changed in migration: stage-based revenue calculations "
            "will use wrong stage mappings, corrupting pipeline forecasts. "
            "(3) MEDIUM — Pipeline last ran 3 days ago; tonight's run is first post-migration; "
            "high risk of schema mismatch causing full pipeline failure at 02:00 UTC. "
            "REMEDIATION: Run field mapping reconciliation NOW; freeze CFO report until validated; "
            "execute dry-run pipeline in staging environment before 22:00 UTC tonight."
        ),

        TaskType.BUSINESS_SIMULATION: (
            "SCENARIO: Switch from Solvent X22 (ChemCo A) to BioSolvent Y11 (GreenChem B). "
            "FINANCIAL: +18% solvent cost = +€432k/year on EUR 2.4M contract; "
            "GMP revalidation cost estimated €800k–€1.2M; total Year 1 impact: €1.2M–€1.6M. "
            "REGULATORY: REACH authorisation avoided; GMP revalidation 12–18 months mandatory — "
            "if started now, compliant by Q1 2026, ahead of Q3 2026 REACH deadline. "
            "ESG: SVHC elimination achieves 2027 ESG target 12 months early; "
            "positive impact on EU Taxonomy alignment and ESG investor ratings. "
            "RECOMMENDATION: GO — phased transition. Initiate GMP revalidation Q1 2025; "
            "qualify BioSolvent Y11 in parallel; complete switchover Q1 2026. "
            "Board approval required for COGS increase >15%."
        ),
    }

    collator = EnterpriseWorldModelDataCollator(tokenizer, max_length=max_length)
    dataset  = []
    for task_type, context in SCENARIOS.items():
        target  = ground_truths[task_type]
        sample  = collator.encode_sample(context, target)
        dataset.append(sample)

    return dataset, collator


def run_training(model, tokenizer, output_dir: str = "./jepa_kg_world_model"):
    """Fine-tune the world model on enterprise task scenarios."""
    console.print(Panel("[bold green]Starting Enterprise World Model Training[/bold green]"))

    dataset, collator = build_training_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = JEPAKGTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        jepa_weight=0.15,
    )

    trainer.train()
    console.print(f"[bold green]Training complete. Model saved to {output_dir}[/bold green]")
    return trainer


# ---------------------------------------------------------------------------
# 11. DEMO RUNNER — ALL FIVE ENTERPRISE TASKS
# ---------------------------------------------------------------------------

def run_demo(model, tokenizer):
    """
    Run inference for all five enterprise world model tasks and display results.
    In zero-shot mode (no fine-tuning), the base LLM reasons from the
    KG-structured prompt and system instructions.
    """
    engine = EnterpriseWorldModelInference(model, tokenizer, max_new_tokens=600)

    task_labels = {
        TaskType.SUPPLY_CHAIN_IMPACT:     "Supply Chain Cascade Prediction",
        TaskType.COMPLIANCE_INFERENCE:    "Compliance Obligation Inference",
        TaskType.PROCESS_VIOLATION:       "Process Constraint Violation Prediction",
        TaskType.DATA_QUALITY_PREDICTION: "Data Quality Defect Prediction",
        TaskType.BUSINESS_SIMULATION:     "Business Impact Simulation",
    }

    table = Table(title="Enterprise World Model — Task Overview", style="bold")
    table.add_column("Task", style="cyan")
    table.add_column("Scenario", style="white")
    for task, label in task_labels.items():
        scenario = SCENARIOS[task]
        table.add_row(label, scenario.observation[:80] + "…")
    console.print(table)
    console.print()

    results = {}
    for task_type, context in SCENARIOS.items():
        label = task_labels[task_type]
        console.print(Panel(
            f"[bold yellow]{label}[/bold yellow]\n\n"
            f"[dim]{context.observation}[/dim]",
            title=f"[bold]Task: {task_type.value}[/bold]",
            border_style="yellow",
        ))

        prediction = engine.predict(context)
        results[task_type] = prediction

        console.print(Panel(
            prediction,
            title="[bold green]World Model Prediction[/bold green]",
            border_style="green",
        ))
        console.print()

    return results


# ---------------------------------------------------------------------------
# 12. ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    console.print(Panel(
        "[bold blue]JEPA-KG Enterprise World Model[/bold blue]\n"
        "Neuro-Symbolic Predictive Intelligence for the Self-Understanding Enterprise\n\n"
        "Tasks:\n"
        "  [1] Supply Chain Cascade Prediction\n"
        "  [2] Compliance Obligation Inference\n"
        "  [3] Process Constraint Violation Prediction\n"
        "  [4] Data Quality Defect Prediction\n"
        "  [5] Business Impact Simulation",
        border_style="blue",
    ))

    # --- Choose mode ----------------------------------------------------------
    # Set TRAIN_MODE = True to fine-tune the model on the enterprise scenarios.
    # Set TRAIN_MODE = False for zero-shot demo using the base model.
    TRAIN_MODE  = False
    MODEL_NAME  = "google/gemma-2-2b-it"   # swap for any HF causal LLM

    model, tokenizer = initialize_world_model(MODEL_NAME)

    if TRAIN_MODE:
        run_training(model, tokenizer)

    run_demo(model, tokenizer)

    # --- Export scenario KG structures as JSON --------------------------------
    export = {}
    for task_type, ctx in SCENARIOS.items():
        export[task_type.value] = {
            "task":        task_type.value,
            "triples":     [{"s": t.subject, "p": t.predicate, "o": t.obj} for t in ctx.triples],
            "observation": ctx.observation,
            "constraints": ctx.constraints,
            "metadata":    ctx.metadata,
        }
    with open("enterprise_scenarios.json", "w") as f:
        json.dump(export, f, indent=2)
    console.print("[dim]Scenario KG structures exported to enterprise_scenarios.json[/dim]")


if __name__ == "__main__":
    main()
