"""
Hetionet Knowledge Graph Evaluation for LLM-JEPA.

Generates responses for the hetionet test set and scores them with entity-level
Precision, Recall, and F1 — because the ground-truth answers are long structured
texts, exact-match scoring is meaningless here.

Metrics computed:
  - Entity Recall@K  : fraction of GT entities recovered in top-K generated lines
  - Entity Precision : fraction of generated entities that are in GT
  - F1               : harmonic mean of precision and recall
  - Priority-section recall (Priority 1 … N separately)

Usage:
  python evaluate_hetionet.py \
    --model_name trained_models/ft-hetionet-jepa-2e-5-0.05-42 \
    --original_model_name meta-llama/Llama-3.2-1B-Instruct \
    --input_file datasets/hetionet_test.jsonl \
    --output_file eval_hetionet.jsonl \
    --max_new_tokens 512

  # Compare against base model:
  python evaluate_hetionet.py \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --original_model_name meta-llama/Llama-3.2-1B-Instruct \
    --input_file datasets/hetionet_test.jsonl \
    --output_file eval_hetionet_base.jsonl \
    --max_new_tokens 512
"""

import argparse
import json
import re
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

# Matches lines like "- Aspirin (Compound)" or "- APOE (Gene)"
_ENTITY_RE = re.compile(r"^-\s+(.+?)\s+\((\w[\w\s]*)\)")

# Matches "... and N more" summary lines — skip these
_MORE_RE = re.compile(r"^\.\.\. and \d+ more")


def extract_entities(text: str) -> set:
    """
    Parse all "- Name (Kind)" lines from a prediction or ground-truth block.
    Returns a set of lowercase "name (kind)" strings for easy set comparison.
    "[already known: X]" lines are included so that GT covers the revealed hints.
    """
    entities = set()
    for line in text.splitlines():
        line = line.strip()
        # Handle "[already known: Name (Kind)]"
        known_match = re.match(r"\[already known:\s+(.+?)\s+\((\w[\w\s]*)\)\]", line)
        if known_match:
            name, kind = known_match.group(1), known_match.group(2)
            entities.add(f"{name.lower()} ({kind.lower()})")
            continue
        m = _ENTITY_RE.match(line)
        if m and not _MORE_RE.match(line):
            name, kind = m.group(1), m.group(2)
            entities.add(f"{name.lower()} ({kind.lower()})")
    return entities


def extract_priority_entities(text: str) -> dict:
    """
    Returns {priority_label: set_of_entities} for each [Priority N] section.
    """
    sections = {}
    current_label = "unknown"
    current_entities = set()

    for line in text.splitlines():
        line = line.strip()
        priority_match = re.match(r"\[Priority \d+\]\s+(.*?)\s+\(", line)
        if priority_match:
            if current_entities:
                sections[current_label] = sections.get(current_label, set()) | current_entities
            current_label = priority_match.group(1).strip()
            current_entities = set()
            continue
        m = _ENTITY_RE.match(line)
        if m and not _MORE_RE.match(line):
            name, kind = m.group(1), m.group(2)
            current_entities.add(f"{name.lower()} ({kind.lower()})")

    if current_entities:
        sections[current_label] = sections.get(current_label, set()) | current_entities
    return sections


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def prf1(predicted: set, ground_truth: set):
    """Compute precision, recall, F1 for two sets of entity strings."""
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not ground_truth:
        return 0.0, 0.0, 0.0
    tp = len(predicted & ground_truth)
    precision = tp / len(predicted)
    recall = tp / len(ground_truth)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Model loading / generation
# ---------------------------------------------------------------------------

def load_model(model_name: str, original_model_name: str, device_map="auto"):
    print(f"Loading tokenizer from: {original_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        original_model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def build_prompt(messages: list, tokenizer, model_name: str) -> str:
    """Build the prompt up to (but not including) the assistant turn."""
    if "google/gemma" in model_name:
        prompt_messages = [
            {"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}
        ]
    else:
        prompt_messages = messages[:2]  # system + user

    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, device) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        do_sample=False,
        repetition_penalty=1.1,
    )
    out = model.generate(**inputs, generation_config=gen_cfg)
    # Decode only the new tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.model_name, args.original_model_name)

    dataset = load_dataset("json", data_files=args.input_file)["train"]
    if args.max_examples:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    print(f"\nEvaluating {len(dataset)} examples from {args.input_file}")
    print(f"Output: {args.output_file}\n")

    # Resume from existing output file
    done_indices = set()
    if os.path.exists(args.output_file) and not args.no_skip_existing:
        with open(args.output_file) as f:
            for line in f:
                rec = json.loads(line)
                if "index" in rec:
                    done_indices.add(rec["index"])
        print(f"Resuming: {len(done_indices)} already done.")

    out_fh = open(args.output_file, "a", encoding="utf-8")

    # Aggregated metrics
    total_p = total_r = total_f1 = 0.0
    per_priority: dict = {}   # priority_label -> [precision, recall, f1]
    n_scored = 0

    for idx, example in enumerate(tqdm(dataset)):
        if idx in done_indices:
            continue

        messages = example["messages"]
        ground_truth_text = messages[2]["content"]

        prompt = build_prompt(messages, tokenizer, args.original_model_name)
        generated = generate(model, tokenizer, prompt, args.max_new_tokens, device)

        gt_entities = extract_entities(ground_truth_text)
        gen_entities = extract_entities(generated)

        p, r, f1 = prf1(gen_entities, gt_entities)
        total_p += p
        total_r += r
        total_f1 += f1
        n_scored += 1

        # Per-priority breakdown
        gt_prio = extract_priority_entities(ground_truth_text)
        gen_prio = extract_priority_entities(generated)
        all_labels = set(gt_prio) | set(gen_prio)
        prio_scores = {}
        for label in all_labels:
            pp, pr, pf = prf1(gen_prio.get(label, set()), gt_prio.get(label, set()))
            prio_scores[label] = {"precision": pp, "recall": pr, "f1": pf}
            if label not in per_priority:
                per_priority[label] = []
            per_priority[label].append((pp, pr, pf))

        record = {
            "index": idx,
            "entity": messages[1]["content"].split("\n")[0],
            "precision": p,
            "recall": r,
            "f1": f1,
            "gt_entity_count": len(gt_entities),
            "gen_entity_count": len(gen_entities),
            "per_priority": prio_scores,
            "generated": generated,
            "ground_truth": ground_truth_text,
        }
        out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_fh.flush()

    out_fh.close()

    if n_scored == 0:
        print("No new examples scored.")
        return

    print("\n" + "=" * 60)
    print(f"OVERALL RESULTS  ({n_scored} examples)")
    print("=" * 60)
    print(f"  Precision : {total_p / n_scored:.4f}")
    print(f"  Recall    : {total_r / n_scored:.4f}")
    print(f"  F1        : {total_f1 / n_scored:.4f}")

    if per_priority:
        print("\nPER-PRIORITY SECTION RECALL:")
        for label, scores in sorted(per_priority.items()):
            avg_r = sum(s[1] for s in scores) / len(scores)
            avg_f1 = sum(s[2] for s in scores) / len(scores)
            print(f"  {label:<45}  recall={avg_r:.3f}  f1={avg_f1:.3f}  (n={len(scores)})")

    # Write summary to a separate file
    summary_path = args.output_file.replace(".jsonl", "_summary.json")
    summary = {
        "model": args.model_name,
        "input_file": args.input_file,
        "n_examples": n_scored,
        "overall": {
            "precision": total_p / n_scored,
            "recall": total_r / n_scored,
            "f1": total_f1 / n_scored,
        },
        "per_priority": {
            label: {
                "recall": sum(s[1] for s in scores) / len(scores),
                "f1": sum(s[2] for s in scores) / len(scores),
                "n": len(scores),
            }
            for label, scores in per_priority.items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned model on the Hetionet KG prediction task."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path to fine-tuned model (or HF hub name for base model).")
    parser.add_argument("--original_model_name", type=str, required=True,
                        help="Original base model name (used for tokenizer & chat template).")
    parser.add_argument("--input_file", type=str, default="datasets/hetionet_test.jsonl",
                        help="JSONL test file.")
    parser.add_argument("--output_file", type=str, default="eval_hetionet.jsonl",
                        help="JSONL output — one scored record per line.")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate per example (default: 512).")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Limit number of test examples (default: all).")
    parser.add_argument("--no_skip_existing", action="store_true",
                        help="Re-evaluate examples already in output file.")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
