#!/usr/bin/env python3
"""
rc_eval_unified.py

Routing Consistency (RC) + Routing Entropy evaluation for VILA-M3 across
BraTS, VISTA3D, and CXR datasets.

How inference works
-------------------
This is purely a routing-decision eval — the expert model NEVER runs.
For each (image, question) pair we feed a single-turn prompt to VILA-M3,
collect its first response (max 64 tokens), and check whether a routing
token appears in the text:

  BraTS   → expects <BRATS()>        in first response
  VISTA3D → expects <VISTA3D(...)>   in first response
  CXR     → expects <CXR()>          in first response

A response with no such token counts as routing to "none".
64 tokens is always enough — the routing decision appears at the start.

Metrics
-------
RC(group) = Σ_k C(n_k, 2) / C(n, 2)
  Fraction of question-pairs that agree on routing.
  1.0 = all paraphrases route identically, 0.0 = all differ.

H(group) = -Σ_k p_k log2(p_k) / log2(4)
  Normalized Shannon entropy over {BRATS, VISTA3D, CXR, none}.
  0.0 = perfectly consistent, 1.0 = maximally scattered.

RC and H are complementary:
  RC=0.5 could mean 50/50 VISTA3D+None  (H ≈ 0.5)
               OR    25% each of 4 experts (H = 1.0, far worse)

Phases
------
  1. samples     Build (image × question) pairs from test items + questions.json
  2. inference   VILA-M3 single-turn greedy decode (JSONL checkpoint, resumable)
  3. metrics     Compute RC + entropy per (item, category), write JSON + CSV

Questions come from questions.json produced by prepare_training_data.py
--phase paraphrases. That file has Gemma-paraphrased variants for every
question category (brats_grade, brats_seg, vista3d_seg, vista3d_what, cxr).

Usage
-----
  python rc_eval_unified.py \\
    --datasets brats,vista3d,cxr \\
    --questions_json /path/to/training_data/questions.json \\
    --brats_slices_dir /path/to/BraTS/vila_m3_slices \\
    --vista3d_meta /path/to/training_data/vista3d_meta.json \\
    --txrv_preds_dir /path/to/training_data/txrv_preds \\
    --cxr_dir /path/to/mimic-cxr-mini \\
    --vila_model MONAI/Llama3-VILA-M3-8B \\
    --out_dir /path/to/rc_eval_unified

Resume after interruption — inference checkpoint is appended, already-done
sample IDs are automatically skipped on re-run.

Compute metrics only from an existing checkpoint:
  python rc_eval_unified.py --metrics_only [same args]
"""

import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

# ── Model cards (verbatim from expert_utils.py) ───────────────────────────────

MODEL_CARDS = (
    "Here is a list of available expert models:\n"
    "<BRATS(args)> Modality: MRI, Task: segmentation, Overview: A pre-trained model for volumetric "
    "(3D) segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, "
    "Accuracy: Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 "
    "- Average: 0.8518, Valid args are: None\n"
    "<VISTA3D(args)> Modality: CT, Task: segmentation, Overview: domain-specialized interactive "
    "foundation model developed for segmenting and annotating human anatomies with precision, "
    "Accuracy: 127 organs: 0.792 Dice on average, Valid args are: 'everything', 'hepatic tumor', "
    "'pancreatic tumor', 'lung tumor', 'bone lesion', 'organs', 'cardiovascular', "
    "'gastrointestinal', 'skeleton', or 'muscles'\n"
    "<VISTA2D(args)> Modality: cell imaging, Task: segmentation, Overview: model for cell "
    "segmentation, which was trained on a variety of cell imaging outputs, including brightfield, "
    "phase-contrast, fluorescence, confocal, or electron microscopy, Accuracy: Good accuracy "
    "across several cell imaging datasets, Valid args are: None\n"
    "<CXR(args)> Modality: chest x-ray (CXR), Task: classification, Overview: pre-trained model "
    "which are trained on large cohorts of data, Accuracy: Good accuracy across several diverse "
    "chest x-rays datasets, Valid args are: None\n"
    "Give the model <NAME(args)> when selecting a suitable expert model.\n"
)

# Prompt headers per dataset — must match training conversation format exactly
MRI_HEADER = (
    MODEL_CARDS
    + "T1(contrast enhanced): <image>, T1: <image>, T2: <image>, FLAIR: <image> "
    "These are different MRI modalities.\n"
)
CT_HEADER  = MODEL_CARDS + " <image>This is a CT image.\n"
CXR_HEADER = MODEL_CARDS + "<image> This is a CXR image.\n"

# Replicates the demo's mentioned_by() check:
#   re.findall(r"<(.*?)>", output) must return exactly 1 match.
# Zero or 2+ angle-bracket tokens → no expert runs (routing = "none").
_ALL_ANGLE_RE = re.compile(r"<(.*?)>")

# Entropy denominator: {BRATS, VISTA3D, CXR, none}
N_EXPERTS = 4

# How routing works per dataset (all single-turn: check GPT turn 1 only):
#
#  brats    → 4 MRI PNGs + question → expected: <BRATS()> in first response
#               grade: "Let me first trigger <BRATS()>."
#               seg:   "I segmented any brain tumors using <BRATS()>."
#
#  vista3d  → 1 CT PNG + question → expected: <VISTA3D(arg)> in first response
#               seg:  "I segmented the liver using <VISTA3D(hepatic tumor)>."
#               what: "This looks like a CT image. Let me trigger <VISTA3D(organs)>. "
#
#  cxr      → 1 CXR JPEG + question → expected: <CXR()> in first response
#               "This looks like a chest x-ray. Let me first trigger <CXR()>."
#
# In all cases the routing token appears in the MODEL'S FIRST RESPONSE.
# We never send the second human turn (expert results) — that's only for training.
DATASET_CONFIGS = {
    "brats": {
        "categories":      ["brats_grade", "brats_seg"],
        "expected_expert": "BRATS",
        "prompt_header":   MRI_HEADER,
    },
    "vista3d": {
        "categories":      ["vista3d_seg", "vista3d_what"],
        "expected_expert": "VISTA3D",
        "prompt_header":   CT_HEADER,
    },
    "cxr": {
        "categories":      ["cxr"],
        "expected_expert": "CXR",
        "prompt_header":   CXR_HEADER,
    },
}


# ── Routing + metric helpers ───────────────────────────────────────────────────

def parse_routing(text: str) -> str:
    """
    Replicate the VILA-M3 demo's mentioned_by() check exactly.
    The demo does: re.findall(r"<(.*?)>", output) and requires EXACTLY 1 match.
    Zero matches → 'none'. Two or more matches → 'none' (no expert would run).
    Expert priority matches demo order: CXR > VISTA3D > BRATS.
    """
    matches = _ALL_ANGLE_RE.findall(text)
    if len(matches) != 1:
        return "none"
    token = matches[0]
    if "CXR" in token:
        return "CXR"
    if "VISTA3D" in token:
        return "VISTA3D"
    if "BRATS" in token:
        return "BRATS"
    return "none"


def compute_rc(routings: list) -> float:
    """RC = Σ_k C(n_k,2) / C(n,2). 1.0 = all identical, 0.0 = all different."""
    n = len(routings)
    if n < 2:
        return 1.0
    agreements = sum(c * (c - 1) // 2 for c in Counter(routings).values())
    return agreements / (n * (n - 1) // 2)


def compute_entropy(routings: list) -> float:
    """Normalized Shannon entropy. 0.0 = all same, 1.0 = maximally spread."""
    n = len(routings)
    if n == 0:
        return 0.0
    counts = Counter(routings)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return h / math.log2(N_EXPERTS)


# ── Test item loading ──────────────────────────────────────────────────────────

def load_brats_test_items(brats_slices_dir: str, n_test: int = 0, seed: int = 42) -> list:
    """Load BraTS test slices from extracted_slices_meta.json (split == 'test').
    n_test=0 uses all test slices."""
    slices_dir = Path(os.path.expanduser(brats_slices_dir))
    meta_path = slices_dir / "extracted_slices_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"BraTS metadata not found: {meta_path}")
    with open(meta_path) as f:
        all_meta = json.load(f)

    items, skipped = [], 0
    for m in all_meta:
        if m.get("split") != "test":
            continue
        images = [str(slices_dir / m["image"][i]) for i in range(4)]
        if not all(Path(p).exists() for p in images):
            skipped += 1
            continue
        subject = Path(images[0]).parent.name
        base = Path(images[0]).name.replace("_img_t1ce.png", "")
        items.append({"item_id": f"{subject}/{base}", "images": images})
    if skipped:
        print(f"  BraTS: {skipped} slices skipped (missing PNGs)")
    if n_test > 0 and len(items) > n_test:
        rng = random.Random(seed)
        items = rng.sample(items, n_test)
    return items


def load_vista3d_test_items(vista3d_meta: str, n_test: int, seed: int = 42) -> list:
    """Sample n_test CT slices from vista3d_meta.json (0 = use all)."""
    with open(os.path.expanduser(vista3d_meta)) as f:
        all_meta = json.load(f)

    rng = random.Random(seed)
    rng.shuffle(all_meta)
    items, skipped = [], 0
    for m in all_meta:
        if not Path(m["image"]).exists():
            skipped += 1
            continue
        stem = Path(m["image"]).stem
        items.append({"item_id": stem, "images": [m["image"]]})
        if n_test > 0 and len(items) >= n_test:
            break
    if skipped:
        print(f"  VISTA3D: {skipped} slices skipped (missing PNGs)")
    return items


def load_cxr_test_items(txrv_preds_dir: str, cxr_dir: str, n_test: int, seed: int = 42) -> list:
    """Sample n_test CXR images from txrv_preds/ (0 = use all)."""
    preds_dir = Path(os.path.expanduser(txrv_preds_dir))
    files_dir = Path(os.path.expanduser(cxr_dir)) / "files"

    jpg_index = {p.stem: p for p in files_dir.rglob("*.jpg")}
    pred_files = sorted(preds_dir.glob("*.json"))
    rng = random.Random(seed)
    rng.shuffle(pred_files)

    items, skipped = [], 0
    for pf in pred_files:
        jpg_path = jpg_index.get(pf.stem)
        if jpg_path is None:
            skipped += 1
            continue
        items.append({"item_id": pf.stem, "images": [str(jpg_path)]})
        if n_test > 0 and len(items) >= n_test:
            break
    if skipped:
        print(f"  CXR: {skipped} pred files had no matching JPEG")
    return items


# ── Eval sample generation ─────────────────────────────────────────────────────

def build_eval_samples(test_items: list, questions: dict, dataset: str) -> list:
    """
    For each test item × question category × paraphrase → one eval sample.
    Each sample has a single-turn conversation: human prompt → empty gpt slot.
    """
    cfg = DATASET_CONFIGS[dataset]
    samples = []
    for item in test_items:
        for cat in cfg["categories"]:
            cat_qs = questions.get(cat, [])
            if not cat_qs:
                print(f"  Warning: no questions for category '{cat}' in questions.json")
                continue
            for q_idx, question in enumerate(cat_qs):
                samples.append({
                    "sample_id":  f"{dataset}__{item['item_id']}__{cat}__{q_idx:03d}",
                    "dataset":    dataset,
                    "item_id":    item["item_id"],
                    "category":   cat,
                    "expected":   cfg["expected_expert"],
                    "q_idx":      q_idx,
                    "question":   question,
                    "images":     item["images"],
                    # Single-turn prompt; gpt slot is empty (routing captured from output)
                    "conversations": [
                        {"from": "human", "value": cfg["prompt_header"] + question},
                        {"from": "gpt",   "value": ""},
                    ],
                })
    return samples


# ── VILA-M3 inference ──────────────────────────────────────────────────────────

def _mock_deepspeed():
    import sys, types, importlib.machinery
    if "deepspeed" in sys.modules:
        return
    ds = types.ModuleType("deepspeed")
    ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    ds.__path__ = []
    ds_comm = types.ModuleType("deepspeed.comm")
    ds_comm.__spec__ = importlib.machinery.ModuleSpec("deepspeed.comm", None)
    for fn in ("is_initialized", "barrier", "all_reduce", "broadcast", "new_group"):
        setattr(ds_comm, fn, lambda *a, **kw: None)
    ds_comm.get_rank = lambda: 0
    ds_comm.get_world_size = lambda: 1
    sys.modules["deepspeed"] = ds
    sys.modules["deepspeed.comm"] = ds_comm


def load_vila(model_path: str, model_base, conv_mode: str):
    _mock_deepspeed()
    import transformers.utils as _tutils
    if not hasattr(_tutils, "is_tf_available"):
        _tutils.is_tf_available = lambda: False

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    expanded = os.path.expanduser(model_path)
    if os.path.isdir(expanded):
        resolved = expanded
        print(f"Loading VILA-M3 from local path: {resolved}")
    else:
        print(f"Downloading VILA-M3 from HuggingFace: {model_path}")
        from huggingface_hub import snapshot_download
        resolved = snapshot_download(model_path)

    disable_torch_init()
    model_name = get_model_name_from_path(resolved)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        resolved, model_name, model_base
    )
    cuda_ok = torch.cuda.is_available() and next(model.parameters()).is_cuda
    print(f"VILA-M3 loaded  name={model_name}  conv={conv_mode}  ctx={context_len}  cuda={cuda_ok}")
    if not cuda_ok:
        print("  WARNING: no GPU — converting to float32 for CPU inference")
        model = model.float()
    return tokenizer, model, image_processor


def run_inference(
    tokenizer, model, image_processor, conv_mode: str,
    samples: list, checkpoint_path: str, max_new_tokens: int,
):
    """
    Single-turn VILA-M3 inference. Appends to JSONL checkpoint; skips done IDs.
    Each result records: sample_id, routing decision, raw output text.
    """
    _mock_deepspeed()
    import transformers.utils as _tutils
    if not hasattr(_tutils, "is_tf_available"):
        _tutils.is_tf_available = lambda: False

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
    from PIL import Image

    done_ids: set = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"  Checkpoint: {len(done_ids):,} done")

    pending = [s for s in samples if s["sample_id"] not in done_ids]
    print(f"  Pending:    {len(pending):,} samples")
    if not pending:
        print("  Nothing to do — all samples already in checkpoint.")
        return

    conv_template = conv_templates[conv_mode]
    raw_stop = (
        conv_template.sep
        if conv_template.sep_style != SeparatorStyle.TWO
        else conv_template.sep2
    )
    stop_str = raw_stop if raw_stop else "<|eot_id|>"

    with open(checkpoint_path, "a") as ckpt_f:
        for sample in tqdm(pending, desc="VILA-M3 inference"):
            try:
                images = [Image.open(p).convert("RGB") for p in sample["images"]]
                human_text = sample["conversations"][0]["value"]

                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], human_text)
                if conv.sep_style == SeparatorStyle.LLAMA_3:
                    conv.append_message(conv.roles[1], "")
                else:
                    conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()
                inf_dtype = torch.float16 if next(model.parameters()).is_cuda else torch.float32
                images_tensor = process_images(images, image_processor, model.config).to(
                    model.device, dtype=inf_dtype
                )
                input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0).to(model.device)
                )
                stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=[images_tensor],
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping],
                        pad_token_id=tokenizer.eos_token_id,
                    )

                new_ids = (
                    output_ids[0]
                    if output_ids.shape[1] <= input_ids.shape[1]
                    else output_ids[0][input_ids.shape[1]:]
                )
                text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                if text.endswith(stop_str):
                    text = text[: -len(stop_str)].strip()

                result = {
                    "sample_id": sample["sample_id"],
                    "dataset":   sample["dataset"],
                    "item_id":   sample["item_id"],
                    "category":  sample["category"],
                    "expected":  sample["expected"],
                    "q_idx":     sample["q_idx"],
                    "question":  sample["question"],
                    "output":    text,
                    "routing":   parse_routing(text),
                }
            except Exception as e:
                result = {
                    "sample_id": sample["sample_id"],
                    "dataset":   sample["dataset"],
                    "item_id":   sample["item_id"],
                    "category":  sample["category"],
                    "expected":  sample["expected"],
                    "q_idx":     sample["q_idx"],
                    "question":  sample["question"],
                    "output":    f"ERROR: {e}",
                    "routing":   "error",
                }

            ckpt_f.write(json.dumps(result) + "\n")
            ckpt_f.flush()


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_global_re(all_routings: list) -> dict:
    """
    Routing Entropy as defined in the paper: a GLOBAL metric across all samples.
    p_k = proportion of times expert k was used across all queries.
    H_norm = -Σ p_k ln(p_k) / ln(K)  where K = number of active experts.
    High H (~1) = model uses all experts evenly (adaptive).
    Low H (~0) = expert collapse — model routes everything to one expert.
    """
    n = len(all_routings)
    if n == 0:
        return {"H_norm": 0.0, "distribution": {}, "n_total": 0}
    counts = Counter(all_routings)
    # K = number of distinct expert labels seen (exclude "none" from K if present,
    # since "none" means no expert fired — still include it in p_k calculation)
    active_experts = [k for k in counts if k != "none"]
    K = max(len(active_experts), 2)  # at least 2 to avoid log(1)=0 division
    h_raw = -sum((c / n) * math.log(c / n) for c in counts.values())
    h_norm = h_raw / math.log(K)
    return {
        "H_norm":       round(h_norm, 6),
        "H_raw":        round(h_raw, 6),
        "K_experts":    K,
        "distribution": {k: round(v / n, 4) for k, v in counts.most_common()},
        "counts":       dict(counts),
        "n_total":      n,
    }


def compute_metrics(checkpoint_path: str, out_dir: str) -> dict:
    results = []
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("routing") not in ("error", None) and "output" in r:
                    r["routing"] = parse_routing(r["output"])  # re-parse for any regex fixes
                    results.append(r)
            except json.JSONDecodeError:
                pass
    print(f"\nLoaded {len(results):,} valid inference results")

    # Group by (dataset, item_id, category) → list of routing decisions
    groups: dict = defaultdict(list)
    for r in results:
        groups[(r["dataset"], r["item_id"], r["category"])].append(r["routing"])

    # Per (item, category): RC only (per-item entropy is not the paper's RE)
    per_item_cat = {}
    for (ds, item_id, cat), routings in groups.items():
        counts = Counter(routings)
        per_item_cat[(ds, item_id, cat)] = {
            "rc":                   compute_rc(routings),
            "n_queries":            len(routings),
            "routing_distribution": dict(counts),
            "majority_routing":     counts.most_common(1)[0][0],
        }

    # Per (dataset, category) aggregate RC + correct route rate
    agg: dict = defaultdict(dict)
    for ds, cfg in DATASET_CONFIGS.items():
        for cat in cfg["categories"]:
            vals = [v for (d, i, c), v in per_item_cat.items() if d == ds and c == cat]
            if not vals:
                continue
            rc_scores = [v["rc"] for v in vals]
            expected  = cfg["expected_expert"]
            n_correct = sum(v["routing_distribution"].get(expected, 0) for v in vals)
            n_total   = sum(v["n_queries"] for v in vals)
            agg[ds][cat] = {
                "mean_rc":            sum(rc_scores) / len(rc_scores),
                "median_rc":          sorted(rc_scores)[len(rc_scores) // 2],
                "correct_route_rate": n_correct / n_total if n_total else 0.0,
                "n_items":            len(vals),
                "n_queries_total":    n_total,
            }

    # ── Global Routing Entropy (paper definition) ──────────────────────────────
    # Per-dataset RE first, then average — gives each dataset equal weight
    # regardless of how many queries it has (apple-to-apple across datasets).
    per_ds_re = {}
    for ds in DATASET_CONFIGS:
        ds_routings = [r["routing"] for r in results if r["dataset"] == ds]
        per_ds_re[ds] = compute_global_re(ds_routings)

    # Global RE = average p_k across datasets, then compute H
    all_experts = {"BRATS", "VISTA3D", "CXR", "none"}
    avg_dist: dict = {}
    ds_list = [ds for ds in DATASET_CONFIGS if ds in per_ds_re and per_ds_re[ds]["n_total"] > 0]
    for expert in all_experts:
        avg_dist[expert] = sum(per_ds_re[ds]["distribution"].get(expert, 0.0) for ds in ds_list) / len(ds_list)
    # Normalise (should already sum to 1 but floating point)
    total = sum(avg_dist.values())
    avg_dist = {k: v / total for k, v in avg_dist.items() if v > 0}
    K = len([k for k in avg_dist if k != "none"]) or 2
    h_raw = -sum(p * math.log(p) for p in avg_dist.values())
    global_re = {
        "H_norm":       round(h_raw / math.log(K), 6),
        "H_raw":        round(h_raw, 6),
        "K_experts":    K,
        "distribution": {k: round(v, 4) for k, v in sorted(avg_dist.items(), key=lambda x: -x[1])},
        "note":         "averaged per-dataset p_k for equal dataset weighting",
    }

    metrics = {
        "global_routing_entropy": global_re,
        "per_dataset_routing_entropy": per_ds_re,
        "per_dataset_category_rc": {ds: dict(cats) for ds, cats in agg.items()},
        "per_item_category": {
            f"{ds}__{item}__{cat}": v
            for (ds, item, cat), v in per_item_cat.items()
        },
    }

    # JSON
    json_path = os.path.join(out_dir, "rc_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON → {json_path}")

    # CSV: one row per (item, category)
    csv_path = os.path.join(out_dir, "rc_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset", "item_id", "category",
            "rc", "n_queries", "majority_routing", "routing_distribution",
        ])
        for (ds, item_id, cat), v in sorted(per_item_cat.items()):
            writer.writerow([
                ds, item_id, cat,
                f"{v['rc']:.6f}",
                v["n_queries"],
                v["majority_routing"],
                json.dumps(v["routing_distribution"]),
            ])
    print(f"Metrics CSV  → {csv_path}")

    # Console summary
    print("\n" + "=" * 74)
    print("  Routing Consistency (RC) — per dataset/category")
    print("=" * 74)
    for ds in DATASET_CONFIGS:
        if ds not in agg:
            continue
        print(f"\n  [{ds.upper()}]  expected = {DATASET_CONFIGS[ds]['expected_expert']}")
        for cat, s in agg[ds].items():
            print(
                f"    {cat:<20s}"
                f"  RC={s['mean_rc']:.4f} (med {s['median_rc']:.4f})"
                f"  correct={s['correct_route_rate']:.4f}"
                f"  n={s['n_items']}"
            )

    print("\n" + "=" * 74)
    print("  Routing Entropy (RE) — global across ALL datasets combined")
    print("  H~1 = adaptive (uses all experts)  H~0 = expert collapse")
    print("=" * 74)
    g = global_re
    print(f"\n  Global  H_norm={g['H_norm']:.4f}  K={g['K_experts']}  n={g['n_total']}")
    print(f"  Distribution: { {k: f'{v:.3f}' for k,v in g['distribution'].items()} }")
    print(f"\n  Per-dataset RE (should collapse to correct expert — low H is OK here):")
    for ds, re in per_ds_re.items():
        print(f"    [{ds.upper():<8s}]  H_norm={re['H_norm']:.4f}  dist={re['distribution']}")
    print("=" * 74)

    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument("--datasets",       default="brats,vista3d,cxr",
                   help="Comma-separated datasets to evaluate")
    p.add_argument("--questions_json", required=True,
                   help="questions.json from prepare_training_data.py --phase paraphrases")

    # VILA-M3
    p.add_argument("--vila_model",     default="MONAI/Llama3-VILA-M3-8B")
    p.add_argument("--model_base",     default=None)
    p.add_argument("--conv_mode",      default="llama_3")
    p.add_argument("--max_new_tokens", type=int, default=64,
                   help="64 tokens is enough to capture any routing token")

    # BraTS
    p.add_argument("--brats_slices_dir",
                   help="BraTS vila_m3_slices/ directory (must contain extracted_slices_meta.json)")
    p.add_argument("--n_test_brats", type=int, default=200,
                   help="Number of BraTS test slices to evaluate (0 = all)")

    # VISTA3D
    p.add_argument("--vista3d_meta",
                   help="vista3d_meta.json from prepare_training_data.py --phase extract")
    p.add_argument("--n_test_vista3d", type=int, default=200,
                   help="Number of VISTA3D test items to evaluate (0 = all)")

    # CXR
    p.add_argument("--txrv_preds_dir",
                   help="txrv_preds/ directory from prepare_training_data.py --phase txrv")
    p.add_argument("--cxr_dir",
                   help="MIMIC-CXR mini root (must contain files/ with JPEGs)")
    p.add_argument("--n_test_cxr",     type=int, default=200,
                   help="Number of CXR test items to evaluate (0 = all)")

    # Output + control
    p.add_argument("--out_dir",            default="./rc_eval_unified")
    p.add_argument("--force_regen_samples", action="store_true",
                   help="Rebuild eval_samples.json even if it already exists")
    p.add_argument("--metrics_only",        action="store_true",
                   help="Skip inference; compute metrics from existing checkpoint")

    args = p.parse_args()
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",")]
    questions = json.load(open(os.path.expanduser(args.questions_json)))

    samples_path    = os.path.join(out_dir, "eval_samples.json")
    checkpoint_path = os.path.join(out_dir, "inference_results.jsonl")

    # ── Phase 1+2: Build eval samples ─────────────────────────────────────────
    if not args.metrics_only and (
        not os.path.exists(samples_path) or args.force_regen_samples
    ):
        all_samples = []
        for ds in datasets:
            if ds == "brats":
                assert args.brats_slices_dir, "--brats_slices_dir required for brats"
                items = load_brats_test_items(args.brats_slices_dir, args.n_test_brats)
            elif ds == "vista3d":
                assert args.vista3d_meta, "--vista3d_meta required for vista3d"
                items = load_vista3d_test_items(args.vista3d_meta, args.n_test_vista3d)
            elif ds == "cxr":
                assert args.txrv_preds_dir and args.cxr_dir, \
                    "--txrv_preds_dir and --cxr_dir required for cxr"
                items = load_cxr_test_items(args.txrv_preds_dir, args.cxr_dir, args.n_test_cxr)
            else:
                raise ValueError(f"Unknown dataset: {ds}")

            print(f"  {ds}: {len(items)} test items")
            samples = build_eval_samples(items, questions, ds)
            print(f"  {ds}: {len(samples)} eval samples  "
                  f"({len(DATASET_CONFIGS[ds]['categories'])} categories × "
                  f"{len(items)} items × ~{len(samples) // max(len(items), 1)} q/item)")
            all_samples.extend(samples)

        with open(samples_path, "w") as f:
            json.dump(all_samples, f)
        print(f"\nEval samples ({len(all_samples):,}) → {samples_path}")
    else:
        print(f"Loading eval samples from {samples_path}")
        with open(samples_path) as f:
            all_samples = json.load(f)
        if not args.metrics_only:
            all_samples = [s for s in all_samples if s["dataset"] in datasets]

    # ── Phase 3: Inference ─────────────────────────────────────────────────────
    if not args.metrics_only:
        print(f"\nTotal eval samples: {len(all_samples):,}")
        tokenizer, model, image_processor = load_vila(
            args.vila_model, args.model_base, args.conv_mode
        )
        run_inference(
            tokenizer, model, image_processor, args.conv_mode,
            all_samples, checkpoint_path, args.max_new_tokens,
        )
        print(f"\nInference checkpoint → {checkpoint_path}")

    # ── Phase 4: Metrics ───────────────────────────────────────────────────────
    print("\n── Computing metrics ────────────────────────────────────────────────")
    if not os.path.exists(checkpoint_path):
        print("No inference results found. Run without --metrics_only first.")
        return
    compute_metrics(checkpoint_path, out_dir)


if __name__ == "__main__":
    main()
