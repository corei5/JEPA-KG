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

Data Ingestion (NEW):
  - CSV files (subject, predicate, object columns + optional metadata)
  - Turtle / TTL files (RDF Knowledge Graph format)
  - N-Triples (.nt) files
  - JSON / JSON-LD files
  - RDF/XML files
  - Auto-detection by file extension

Requirements:
  pip install torch transformers peft accelerate bitsandbytes datasets rich rdflib pandas

Usage:
  python enterprise_world_model.py
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
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

# RDFLib for TTL / N-Triples / JSON-LD / RDF-XML
try:
    from rdflib import Graph as RDFGraph, URIRef, Literal, BNode
    from rdflib.namespace import RDF, RDFS, OWL
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False
    logging.warning(
        "rdflib not found. TTL/N-Triples/RDF-XML/JSON-LD ingestion disabled. "
        "Install with: pip install rdflib"
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
    subject:   str
    predicate: str
    obj:       str

    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.obj})"


@dataclass
class EnterpriseContext:
    """Structured enterprise context fed into the world model."""
    task_type:   TaskType
    triples:     List[KGTriple]
    observation: str
    constraints: List[str] = field(default_factory=list)
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> str:
        kg_tok = MODALITY_TOKENS
        kg_block        = " ".join(str(t) for t in self.triples)
        constraint_block = "; ".join(self.constraints) if self.constraints else "none"
        return (
            f"{kg_tok['kg_start']} {kg_block} {kg_tok['kg_end']} "
            f"{kg_tok['observation']} {self.observation} "
            f"[constraints: {constraint_block}] "
            f"{' '.join(PREDICTOR_TOKENS[:4])}"
        )


# ---------------------------------------------------------------------------
# 4. DATA PROCESSING — MULTI-FORMAT INGESTION  (NEW)
# ---------------------------------------------------------------------------

class KGFileFormat(str, Enum):
    CSV      = "csv"
    TURTLE   = "ttl"
    NTRIPLES = "nt"
    JSONLD   = "jsonld"
    RDFXML   = "xml"
    JSON     = "json"
    AUTO     = "auto"


# Mapping file extension → KGFileFormat
_EXT_MAP: Dict[str, KGFileFormat] = {
    ".csv":     KGFileFormat.CSV,
    ".ttl":     KGFileFormat.TURTLE,
    ".turtle":  KGFileFormat.TURTLE,
    ".nt":      KGFileFormat.NTRIPLES,
    ".jsonld":  KGFileFormat.JSONLD,
    ".json":    KGFileFormat.JSON,
    ".xml":     KGFileFormat.RDFXML,
    ".rdf":     KGFileFormat.RDFXML,
    ".owl":     KGFileFormat.RDFXML,
}


def _clean_uri(value: str) -> str:
    """
    Strip URI brackets and namespace prefixes to produce human-readable labels.
    E.g.  <http://example.org/ontology#Supplier_A>  →  Supplier_A
          "6 weeks"^^xsd:string                     →  6_weeks
    """
    # Remove angle brackets
    value = value.strip().lstrip("<").rstrip(">")
    # Keep only the local name after # or /
    for sep in ("#", "/"):
        if sep in value:
            value = value.rsplit(sep, 1)[-1]
    # Remove datatype annotations  "literal"^^type
    if "^^" in value:
        value = value.split("^^")[0]
    # Strip surrounding quotes
    value = value.strip('"\'')
    # Replace whitespace with underscore
    value = re.sub(r"\s+", "_", value)
    return value if value else "unknown"


def _node_label(node: Any) -> str:
    """Convert rdflib node to clean string label."""
    if isinstance(node, Literal):
        raw = str(node)
        return re.sub(r"\s+", "_", raw.strip())
    if isinstance(node, (URIRef, BNode)):
        return _clean_uri(str(node))
    return _clean_uri(str(node))


# ── CSV ─────────────────────────────────────────────────────────────────────

def _load_csv(
    source: Union[str, Path, io.StringIO],
    subject_col:   str = "subject",
    predicate_col: str = "predicate",
    object_col:    str = "object",
) -> Tuple[List[KGTriple], Dict[str, Any]]:
    """
    Load KG triples from a CSV file or StringIO buffer.

    Expected columns (configurable):
        subject | predicate | object
        --------+-----------+-------
        SupplierA | supplies | ComponentX

    Optional extra columns are collected as metadata.
    The function is tolerant of:
      - Missing headers  (falls back to positional columns 0,1,2)
      - Extra whitespace in cells
      - BOM characters
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source, encoding="utf-8-sig", skipinitialspace=True)
    else:
        df = pd.read_csv(source, encoding="utf-8-sig", skipinitialspace=True)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Map expected column names to actual
    col_aliases = {
        "subject":   ["subject", "s", "src", "source", "from"],
        "predicate": ["predicate", "p", "relation", "rel", "edge", "property"],
        "object":    ["object", "o", "obj", "target", "dst", "to", "value"],
    }

    def resolve(aliases: List[str], fallback_idx: int) -> str:
        for alias in aliases:
            if alias in df.columns:
                return alias
        if fallback_idx < len(df.columns):
            return df.columns[fallback_idx]
        raise ValueError(
            f"Cannot find a column matching any of {aliases} in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    s_col = resolve(col_aliases["subject"],   0)
    p_col = resolve(col_aliases["predicate"], 1)
    o_col = resolve(col_aliases["object"],    2)

    # Extra columns → metadata list
    extra_cols = [c for c in df.columns if c not in (s_col, p_col, o_col)]
    metadata: Dict[str, Any] = {}
    if extra_cols:
        metadata["csv_extra_columns"] = extra_cols
        metadata["csv_rows"] = len(df)
        for col in extra_cols:
            metadata[col] = df[col].dropna().unique().tolist()

    triples: List[KGTriple] = []
    for _, row in df.iterrows():
        s = _clean_uri(str(row[s_col]))
        p = _clean_uri(str(row[p_col]))
        o = _clean_uri(str(row[o_col]))
        if s and p and o and s != "nan":
            triples.append(KGTriple(s, p, o))

    console.print(
        f"[green]CSV loaded:[/green] {len(triples)} triples "
        f"from columns [{s_col}, {p_col}, {o_col}]"
    )
    return triples, metadata


# ── RDF FORMATS (TTL / N-Triples / JSON-LD / RDF-XML) ───────────────────────

def _load_rdf(
    source: Union[str, Path],
    fmt: KGFileFormat,
    max_triples: int = 5000,
    exclude_rdf_schema: bool = True,
) -> Tuple[List[KGTriple], Dict[str, Any]]:
    """
    Load KG triples from any RDF serialisation supported by rdflib:
      - Turtle (.ttl)
      - N-Triples (.nt)
      - JSON-LD (.jsonld)
      - RDF/XML (.xml / .rdf / .owl)

    Parameters
    ----------
    source : file path
    fmt    : KGFileFormat enum
    max_triples : cap to avoid OOM on huge ontologies
    exclude_rdf_schema : skip RDF/RDFS/OWL schema triples (type, subClassOf, …)
                         to keep the KG focused on instance data
    """
    if not RDFLIB_AVAILABLE:
        raise ImportError(
            "rdflib is required for RDF ingestion. "
            "Install: pip install rdflib"
        )

    rdflib_format_map = {
        KGFileFormat.TURTLE:   "turtle",
        KGFileFormat.NTRIPLES: "nt",
        KGFileFormat.JSONLD:   "json-ld",
        KGFileFormat.RDFXML:   "xml",
    }
    rdf_fmt = rdflib_format_map.get(fmt, "turtle")

    g = RDFGraph()
    g.parse(str(source), format=rdf_fmt)
    console.print(f"[green]RDF parsed:[/green] {len(g)} raw triples ({rdf_fmt})")

    # Schema predicates to skip when exclude_rdf_schema=True
    schema_predicates = {
        str(RDF.type), str(RDFS.subClassOf), str(RDFS.domain),
        str(RDFS.range), str(RDFS.label), str(RDFS.comment),
        str(OWL.equivalentClass), str(OWL.disjointWith),
    } if exclude_rdf_schema else set()

    triples: List[KGTriple] = []
    for s, p, o in g:
        if str(p) in schema_predicates:
            continue
        subject   = _node_label(s)
        predicate = _node_label(p)
        obj       = _node_label(o)
        if subject and predicate and obj:
            triples.append(KGTriple(subject, predicate, obj))
        if len(triples) >= max_triples:
            console.print(
                f"[yellow]Warning:[/yellow] max_triples={max_triples} reached — "
                "truncating. Pass a higher max_triples if needed."
            )
            break

    metadata: Dict[str, Any] = {
        "rdf_format":       rdf_fmt,
        "raw_triple_count": len(g),
        "loaded_triples":   len(triples),
        "namespaces":       {p: str(n) for p, n in g.namespaces()},
    }
    console.print(f"[green]RDF loaded:[/green] {len(triples)} instance triples retained")
    return triples, metadata


# ── JSON (plain / custom schema) ─────────────────────────────────────────────

def _load_json(
    source: Union[str, Path, dict],
    triple_key: str = "triples",
) -> Tuple[List[KGTriple], Dict[str, Any]]:
    """
    Load triples from a plain JSON file.

    Supported schemas:
      A) {"triples": [{"s": "...", "p": "...", "o": "..."}, ...]}
      B) {"triples": [{"subject": "...", "predicate": "...", "object": "..."}, ...]}
      C) {"entities": [...], "relations": [...]}  (graph export format)
      D) Flat list of triple dicts at top level

    If the file looks like JSON-LD (has "@context"), route to rdflib instead.
    """
    if isinstance(source, dict):
        data = source
    else:
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Route to rdflib if this is JSON-LD
    if isinstance(data, dict) and "@context" in data:
        console.print("[dim]JSON-LD detected — routing to rdflib parser[/dim]")
        return _load_rdf(source, KGFileFormat.JSONLD)

    triples: List[KGTriple] = []

    # Schema A/B: top-level "triples" key
    raw_list: Optional[List[dict]] = None
    if isinstance(data, dict) and triple_key in data:
        raw_list = data[triple_key]
    elif isinstance(data, list):
        raw_list = data
    # Schema C: graph export with "entities"/"relations" keys
    elif isinstance(data, dict) and "relations" in data:
        raw_list = data["relations"]

    if raw_list is None:
        raise ValueError(
            f"Cannot interpret JSON structure. "
            f"Expected a list or a dict with key '{triple_key}'. "
            f"Top-level keys found: {list(data.keys()) if isinstance(data, dict) else 'list'}"
        )

    s_aliases = ["s", "subject", "src",  "source", "from", "head"]
    p_aliases = ["p", "predicate", "rel", "relation", "edge", "type"]
    o_aliases = ["o", "object",  "obj",  "target",  "dst",  "to",  "tail", "value"]

    def pick(d: dict, aliases: List[str]) -> str:
        for a in aliases:
            if a in d:
                return str(d[a])
        return ""

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        s = _clean_uri(pick(item, s_aliases))
        p = _clean_uri(pick(item, p_aliases))
        o = _clean_uri(pick(item, o_aliases))
        if s and p and o:
            triples.append(KGTriple(s, p, o))

    # Collect remaining top-level keys as metadata
    metadata: Dict[str, Any] = {
        k: v for k, v in (data.items() if isinstance(data, dict) else {}.items())
        if k not in (triple_key, "relations", "entities", "@context", "@graph")
        and not isinstance(v, (list, dict))
    }
    console.print(f"[green]JSON loaded:[/green] {len(triples)} triples")
    return triples, metadata


# ── MAIN PUBLIC API ─────────────────────────────────────────────────────────

def load_kg_from_file(
    filepath:             Union[str, Path],
    fmt:                  KGFileFormat = KGFileFormat.AUTO,
    task_type:            TaskType     = TaskType.SUPPLY_CHAIN_IMPACT,
    observation:          str          = "",
    constraints:          Optional[List[str]] = None,
    extra_metadata:       Optional[Dict[str, Any]] = None,
    # CSV-specific
    csv_subject_col:      str = "subject",
    csv_predicate_col:    str = "predicate",
    csv_object_col:       str = "object",
    # RDF-specific
    rdf_max_triples:      int  = 5000,
    rdf_exclude_schema:   bool = True,
) -> EnterpriseContext:
    """
    Universal entry point: load a KG from any supported file format and return
    a ready-to-use EnterpriseContext for the JEPA-KG world model.

    Parameters
    ----------
    filepath          : path to the data file
    fmt               : KGFileFormat enum (AUTO = detect from extension)
    task_type         : which enterprise task this context belongs to
    observation       : human-readable description of the situation
    constraints       : list of business/regulatory constraints
    extra_metadata    : additional metadata dict to merge
    csv_subject_col   : CSV column name for subject  (default "subject")
    csv_predicate_col : CSV column name for predicate (default "predicate")
    csv_object_col    : CSV column name for object   (default "object")
    rdf_max_triples   : max triples to load from RDF (avoids OOM)
    rdf_exclude_schema: skip RDF/OWL schema triples

    Returns
    -------
    EnterpriseContext ready for serialization and inference.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Auto-detect format
    if fmt == KGFileFormat.AUTO:
        ext = path.suffix.lower()
        fmt = _EXT_MAP.get(ext)
        if fmt is None:
            raise ValueError(
                f"Cannot auto-detect format from extension '{ext}'. "
                f"Supported: {list(_EXT_MAP.keys())}. "
                "Pass fmt= explicitly."
            )
        console.print(f"[dim]Auto-detected format: {fmt.value}[/dim]")

    # Dispatch
    if fmt == KGFileFormat.CSV:
        triples, file_meta = _load_csv(
            path,
            subject_col=csv_subject_col,
            predicate_col=csv_predicate_col,
            object_col=csv_object_col,
        )
    elif fmt in (KGFileFormat.TURTLE, KGFileFormat.NTRIPLES,
                 KGFileFormat.RDFXML, KGFileFormat.JSONLD):
        triples, file_meta = _load_rdf(path, fmt, rdf_max_triples, rdf_exclude_schema)
    elif fmt == KGFileFormat.JSON:
        triples, file_meta = _load_json(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    # Validate
    if not triples:
        raise ValueError(f"No valid triples extracted from {filepath}. "
                         "Check column names / file structure.")

    # Auto-generate observation if not provided
    if not observation:
        observation = (
            f"Enterprise Knowledge Graph loaded from {path.name}. "
            f"{len(triples)} triples ingested covering "
            f"{len({t.subject for t in triples})} entities."
        )

    # Merge metadata
    metadata: Dict[str, Any] = {
        "source_file": str(path),
        "file_format": fmt.value,
        "triple_count": len(triples),
    }
    metadata.update(file_meta)
    if extra_metadata:
        metadata.update(extra_metadata)

    ctx = EnterpriseContext(
        task_type=task_type,
        triples=triples,
        observation=observation,
        constraints=constraints or [],
        metadata=metadata,
    )
    console.print(
        f"[bold green]EnterpriseContext built:[/bold green] "
        f"{len(triples)} triples | task={task_type.value}"
    )
    return ctx


def load_kg_from_string(
    content:    str,
    fmt:        KGFileFormat,
    task_type:  TaskType = TaskType.SUPPLY_CHAIN_IMPACT,
    observation: str = "",
    constraints: Optional[List[str]] = None,
) -> EnterpriseContext:
    """
    Load a KG directly from a string (CSV text, Turtle text, or JSON text).
    Useful for in-memory / API scenarios.
    """
    if fmt == KGFileFormat.CSV:
        buf = io.StringIO(content)
        triples, file_meta = _load_csv(buf)
    elif fmt == KGFileFormat.JSON:
        data = json.loads(content)
        triples, file_meta = _load_json(data)
    elif fmt in (KGFileFormat.TURTLE, KGFileFormat.NTRIPLES,
                 KGFileFormat.RDFXML, KGFileFormat.JSONLD):
        if not RDFLIB_AVAILABLE:
            raise ImportError("rdflib required for RDF string parsing. pip install rdflib")
        rdflib_fmt_map = {
            KGFileFormat.TURTLE:   "turtle",
            KGFileFormat.NTRIPLES: "nt",
            KGFileFormat.JSONLD:   "json-ld",
            KGFileFormat.RDFXML:   "xml",
        }
        g = RDFGraph()
        g.parse(data=content, format=rdflib_fmt_map[fmt])
        triples = [
            KGTriple(_node_label(s), _node_label(p), _node_label(o))
            for s, p, o in g
        ]
        file_meta = {"raw_triple_count": len(g)}
    else:
        raise ValueError(f"Unsupported format for string loading: {fmt}")

    if not observation:
        observation = (
            f"Knowledge Graph loaded from in-memory string. "
            f"{len(triples)} triples."
        )

    return EnterpriseContext(
        task_type=task_type,
        triples=triples,
        observation=observation,
        constraints=constraints or [],
        metadata={**file_meta, "source": "in-memory-string", "format": fmt.value},
    )


def export_triples_to_csv(ctx: EnterpriseContext, output_path: Union[str, Path]) -> None:
    """Export triples from an EnterpriseContext back to CSV."""
    path = Path(output_path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "predicate", "object"])
        for t in ctx.triples:
            writer.writerow([t.subject, t.predicate, t.obj])
    console.print(f"[green]Exported {len(ctx.triples)} triples → {path}[/green]")


def export_triples_to_ttl(ctx: EnterpriseContext, output_path: Union[str, Path]) -> None:
    """Export triples from an EnterpriseContext to Turtle (.ttl) format."""
    if not RDFLIB_AVAILABLE:
        raise ImportError("rdflib required for TTL export. pip install rdflib")
    path = Path(output_path)
    base = "http://enterprise-world-model.org/kg/"
    g = RDFGraph()
    g.bind("ewm", base)
    for t in ctx.triples:
        s = URIRef(f"{base}{t.subject.replace(' ', '_')}")
        p = URIRef(f"{base}{t.predicate.replace(' ', '_')}")
        o = URIRef(f"{base}{t.obj.replace(' ', '_')}")
        g.add((s, p, o))
    g.serialize(destination=str(path), format="turtle")
    console.print(f"[green]Exported {len(ctx.triples)} triples → {path} (Turtle)[/green]")


# ---------------------------------------------------------------------------
# 5. PREDEFINED ENTERPRISE SCENARIOS (unchanged from original)
# ---------------------------------------------------------------------------

SCENARIOS: Dict[TaskType, EnterpriseContext] = {

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

    TaskType.PROCESS_VIOLATION: EnterpriseContext(
        task_type=TaskType.PROCESS_VIOLATION,
        triples=[
            KGTriple("Process_LoanApproval",      "step_1",       "KYC_Identity_Verification"),
            KGTriple("Process_LoanApproval",      "step_2",       "Credit_Score_Check"),
            KGTriple("Process_LoanApproval",      "step_3",       "Affordability_Assessment"),
            KGTriple("Process_LoanApproval",      "step_4",       "Risk_Scoring"),
            KGTriple("Process_LoanApproval",      "step_5",       "Approval_Decision"),
            KGTriple("KYC_Identity_Verification", "SLA",          "24h"),
            KGTriple("Credit_Score_Check",        "externalAPI",  "Experian_API"),
            KGTriple("Experian_API",              "availability", "94_percent"),
            KGTriple("Affordability_Assessment",  "requires",     "Income_Verification_Doc"),
            KGTriple("Risk_Scoring",              "model",        "ML_Model_v3_2"),
            KGTriple("ML_Model_v3_2",             "drift_status", "high_drift_detected"),
            KGTriple("Process_LoanApproval",      "volume_today", "4200_applications"),
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

    TaskType.DATA_QUALITY_PREDICTION: EnterpriseContext(
        task_type=TaskType.DATA_QUALITY_PREDICTION,
        triples=[
            KGTriple("Pipeline_CRM_to_DWH",       "source",        "Salesforce_CRM"),
            KGTriple("Pipeline_CRM_to_DWH",       "target",        "Snowflake_DWH"),
            KGTriple("Pipeline_CRM_to_DWH",       "schedule",      "daily_02:00_UTC"),
            KGTriple("Salesforce_CRM",            "recentChange",  "Custom_Field_Migration_v4"),
            KGTriple("Custom_Field_Migration_v4", "status",        "completed_yesterday"),
            KGTriple("Pipeline_CRM_to_DWH",       "lastRun",       "success_3_days_ago"),
            KGTriple("Snowflake_DWH",             "downstream",    "Revenue_Dashboard"),
            KGTriple("Revenue_Dashboard",         "usedBy",        "CFO_Board_Report"),
            KGTriple("CFO_Board_Report",          "scheduledAt",   "tomorrow_09:00"),
            KGTriple("Field_AccountRevenue",      "mappingStatus", "unmapped_after_migration"),
            KGTriple("Field_OpportunityStage",    "valueSet",      "changed_in_migration"),
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

    TaskType.BUSINESS_SIMULATION: EnterpriseContext(
        task_type=TaskType.BUSINESS_SIMULATION,
        triples=[
            KGTriple("Supplier_ChemCo_A",       "supplies",      "Solvent_X22"),
            KGTriple("Solvent_X22",             "usedIn",        "Manufacturing_Process_P3"),
            KGTriple("Solvent_X22",             "REACHStatus",   "SVHC_Candidate"),
            KGTriple("Supplier_ChemCo_A",       "contractValue", "EUR_2.4M_annual"),
            KGTriple("Supplier_GreenChem_B",    "offers",        "BioSolvent_Y11"),
            KGTriple("BioSolvent_Y11",          "REACHStatus",   "fully_compliant"),
            KGTriple("BioSolvent_Y11",          "pricePremium",  "18_percent"),
            KGTriple("BioSolvent_Y11",          "qualStatus",    "not_yet_qualified"),
            KGTriple("Manufacturing_Process_P3","output",        "Product_Line_Pharma_API"),
            KGTriple("Product_Line_Pharma_API", "annualRevenue", "EUR_28M"),
            KGTriple("EU_REACH_Restriction",    "effectiveDate", "2026_Q3"),
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
# 6. SYSTEM PROMPTS PER TASK
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: Dict[TaskType, str] = {

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
# 7. MODEL INITIALISATION
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
# 8. DATA COLLATOR — ENTERPRISE MULTI-TASK
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
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def encode_sample(self, context: EnterpriseContext, target: str) -> dict:
        full_text = f"{context.serialize()} {MODALITY_TOKENS['impact']} {target}"
        encoding  = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        context_text = context.serialize()
        context_enc  = self.tokenizer(
            context_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        context_len = min(context_enc["input_ids"].shape[1] - 1, self.max_length - 2)
        target_len  = min(encoding["input_ids"].shape[1]   - 1, self.max_length - 2)

        return {
            "input_ids":       encoding["input_ids"].squeeze(0),
            "attention_mask":  encoding["attention_mask"].squeeze(0),
            "context_end_idx": context_len,
            "target_end_idx":  target_len,
        }

    def __call__(self, examples: List[dict]) -> dict:
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
# 9. JEPA-KG TRAINER
# ---------------------------------------------------------------------------

class JEPAKGTrainer(Trainer):
    """
    Dual-objective trainer:

      Loss = λ_lm   * CrossEntropy(token prediction)
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
        labels       = inputs.pop("labels")
        context_ends = inputs.pop("context_end_idx")
        target_ends  = inputs.pop("target_end_idx")

        outputs = model(
            **inputs,
            labels=labels,
            output_hidden_states=True,
        )

        lm_loss = outputs.loss
        hidden  = outputs.hidden_states[-1]   # (batch, seq_len, hidden_dim)

        B = hidden.shape[0]
        jepa_losses = []
        for i in range(B):
            c_idx = min(
                context_ends[i] if isinstance(context_ends, list) else int(context_ends),
                hidden.shape[1] - 1,
            )
            t_idx = min(
                target_ends[i]  if isinstance(target_ends,  list) else int(target_ends),
                hidden.shape[1] - 1,
            )
            z_ctx = hidden[i, c_idx, :]
            z_tgt = hidden[i, t_idx, :]
            sim   = F.cosine_similarity(z_ctx.unsqueeze(0), z_tgt.unsqueeze(0))
            jepa_losses.append(1.0 - sim)

        jepa_loss  = torch.stack(jepa_losses).mean()
        total_loss = lm_loss + self.jepa_weight * jepa_loss

        return (total_loss, outputs) if return_outputs else total_loss


# ---------------------------------------------------------------------------
# 10. INFERENCE ENGINE
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
# 11. TRAINING PIPELINE
# ---------------------------------------------------------------------------

def build_training_dataset(
    tokenizer,
    max_length: int = 256,
) -> Tuple[List[dict], EnterpriseWorldModelDataCollator]:
    """
    Build a minimal multi-task training dataset from the five scenario templates.
    In production, replace ground-truth targets with real enterprise outcomes
    extracted from Corporate Memory / historical KG snapshots.

    You can also build datasets from external files:
        ctx = load_kg_from_file("supply_chain.ttl", task_type=TaskType.SUPPLY_CHAIN_IMPACT)
        sample = collator.encode_sample(ctx, ground_truth_text)
    """

    ground_truths: Dict[TaskType, str] = {

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
        target = ground_truths[task_type]
        sample = collator.encode_sample(context, target)
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
# 12. DEMO RUNNER — ALL FIVE ENTERPRISE TASKS
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
    table.add_column("Task",     style="cyan")
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
# 13. FILE-BASED DEMO  (NEW)
# ---------------------------------------------------------------------------

def run_file_demo():
    """
    Demonstrate the data processing layer by creating small example files
    in each supported format, loading them as EnterpriseContext objects,
    and printing the serialized result — WITHOUT requiring a GPU or model.
    """
    console.print(Panel(
        "[bold magenta]Data Processing Demo — Multi-Format KG Ingestion[/bold magenta]\n"
        "Demonstrates CSV, JSON, TTL, and N-Triples loading.",
        border_style="magenta",
    ))

    # ── 1. CSV example ─────────────────────────────────────────────────────
    csv_content = textwrap.dedent("""\
        subject,predicate,object,domain
        Supplier_Alpha,supplies,Chip_XZ9,electronics
        Chip_XZ9,usedIn,PCB_Module_A,electronics
        PCB_Module_A,partOf,Server_Platform_Q,datacentre
        Supplier_Alpha,delayedBy,4_weeks,logistics
        Chip_XZ9,stockLevel,2_days_supply,inventory
    """)
    csv_path = Path("/tmp/demo_supply_chain.csv")
    csv_path.write_text(csv_content, encoding="utf-8")

    ctx_csv = load_kg_from_file(
        csv_path,
        task_type=TaskType.SUPPLY_CHAIN_IMPACT,
        observation="Supplier Alpha reports a 4-week delay on Chip XZ9.",
        constraints=["JIT policy: max 3-day buffer", "Penalty clause: $80k/day"],
    )
    console.print("[bold]CSV Context serialized:[/bold]")
    console.print(ctx_csv.serialize()[:300] + " …\n")

    # ── 2. JSON example ────────────────────────────────────────────────────
    json_content = json.dumps({
        "domain": "financial_services",
        "triples": [
            {"s": "Process_CreditApproval", "p": "step_1", "o": "KYC_Check"},
            {"s": "Process_CreditApproval", "p": "step_2", "o": "Score_Model"},
            {"s": "Score_Model",            "p": "drift_status", "o": "high"},
            {"s": "KYC_Check",              "p": "SLA",          "o": "24h"},
        ],
    }, indent=2)
    json_path = Path("/tmp/demo_process.json")
    json_path.write_text(json_content, encoding="utf-8")

    ctx_json = load_kg_from_file(
        json_path,
        task_type=TaskType.PROCESS_VIOLATION,
        observation="Credit approval process with drifted scoring model.",
        constraints=["FCA 72h rule", "GDPR explainability"],
    )
    console.print("[bold]JSON Context serialized:[/bold]")
    console.print(ctx_json.serialize()[:300] + " …\n")

    # ── 3. Turtle (TTL) example ────────────────────────────────────────────
    if RDFLIB_AVAILABLE:
        ttl_content = textwrap.dedent("""\
            @prefix ewm: <http://enterprise-world-model.org/kg/> .

            ewm:Supplier_GreenChem   ewm:supplies        ewm:BioSolvent_Y11 .
            ewm:BioSolvent_Y11       ewm:REACHStatus     ewm:fully_compliant .
            ewm:BioSolvent_Y11       ewm:pricePremium    ewm:18_percent .
            ewm:Manufacturing_P3     ewm:uses            ewm:Solvent_X22 .
            ewm:Solvent_X22          ewm:REACHStatus     ewm:SVHC_Candidate .
            ewm:EU_REACH_Restriction ewm:effectiveDate   ewm:2026_Q3 .
        """)
        ttl_path = Path("/tmp/demo_simulation.ttl")
        ttl_path.write_text(ttl_content, encoding="utf-8")

        ctx_ttl = load_kg_from_file(
            ttl_path,
            task_type=TaskType.BUSINESS_SIMULATION,
            observation="REACH restriction incoming. Evaluating BioSolvent Y11 switch.",
            constraints=["REACH SVHC deadline Q3 2026", "GMP revalidation 12-18 months"],
        )
        console.print("[bold]TTL Context serialized:[/bold]")
        console.print(ctx_ttl.serialize()[:300] + " …\n")

        # ── 4. N-Triples example ───────────────────────────────────────────
        nt_content = textwrap.dedent("""\
            <http://ewm.org/Pipeline_CRM_DWH> <http://ewm.org/source> <http://ewm.org/Salesforce_CRM> .
            <http://ewm.org/Pipeline_CRM_DWH> <http://ewm.org/target> <http://ewm.org/Snowflake_DWH> .
            <http://ewm.org/Salesforce_CRM>   <http://ewm.org/recentChange> <http://ewm.org/Field_Migration_v4> .
            <http://ewm.org/Field_Migration_v4> <http://ewm.org/status> <http://ewm.org/completed_yesterday> .
            <http://ewm.org/Field_AccountRevenue> <http://ewm.org/mappingStatus> <http://ewm.org/unmapped> .
        """)
        nt_path = Path("/tmp/demo_data_quality.nt")
        nt_path.write_text(nt_content, encoding="utf-8")

        ctx_nt = load_kg_from_file(
            nt_path,
            task_type=TaskType.DATA_QUALITY_PREDICTION,
            observation="CRM→DWH pipeline post-migration. AccountRevenue field unmapped.",
            constraints=["DWH SLA: fresh by 06:00 UTC", "Board report zero-tolerance"],
        )
        console.print("[bold]N-Triples Context serialized:[/bold]")
        console.print(ctx_nt.serialize()[:300] + " …\n")

    else:
        console.print("[yellow]rdflib not installed — skipping TTL and N-Triples demo.[/yellow]")

    # ── Export round-trip ──────────────────────────────────────────────────
    export_triples_to_csv(ctx_csv, "/tmp/demo_export_roundtrip.csv")
    if RDFLIB_AVAILABLE:
        export_triples_to_ttl(ctx_csv, "/tmp/demo_export_roundtrip.ttl")

    console.print(Panel(
        "[green]Data processing demo complete.[/green]\n"
        "All EnterpriseContext objects are ready for model inference or training.",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# 14. ENTRY POINT
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
        "  [5] Business Impact Simulation\n\n"
        "Data Ingestion:\n"
        "  CSV | Turtle (TTL) | N-Triples | JSON / JSON-LD | RDF/XML",
        border_style="blue",
    ))

    # ── Configuration ──────────────────────────────────────────────────────
    # Set TRAIN_MODE = True  to fine-tune the model on the enterprise scenarios.
    # Set TRAIN_MODE = False for zero-shot demo using the base model.
    # Set FILE_DEMO  = True  to run the multi-format data ingestion demo
    #                        (does NOT require a GPU or model download).
    TRAIN_MODE = False
    FILE_DEMO  = True          # ← flip to False to skip file demo
    MODEL_NAME = "google/gemma-2-2b-it"

    # ── Optional: load a custom KG from file before running inference ──────
    # Uncomment and adjust path/task to use your own data:
    #
    # custom_ctx = load_kg_from_file(
    #     "my_supply_chain.ttl",
    #     task_type=TaskType.SUPPLY_CHAIN_IMPACT,
    #     observation="...",
    #     constraints=["...", "..."],
    # )
    # SCENARIOS[TaskType.SUPPLY_CHAIN_IMPACT] = custom_ctx

    if FILE_DEMO:
        run_file_demo()

    model, tokenizer = initialize_world_model(MODEL_NAME)

    if TRAIN_MODE:
        run_training(model, tokenizer)

    run_demo(model, tokenizer)

    # ── Export scenario KG structures as JSON ──────────────────────────────
    export: Dict[str, Any] = {}
    for task_type, ctx in SCENARIOS.items():
        export[task_type.value] = {
            "task":        task_type.value,
            "triples":     [{"s": t.subject, "p": t.predicate, "o": t.obj} for t in ctx.triples],
            "observation": ctx.observation,
            "constraints": ctx.constraints,
            "metadata":    ctx.metadata,
        }
    with open("enterprise_scenarios.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    console.print("[dim]Scenario KG structures exported to enterprise_scenarios.json[/dim]")


if __name__ == "__main__":
    main()
