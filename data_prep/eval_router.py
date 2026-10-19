#!/usr/bin/env python3
"""
eval_router.py

Evaluates the decoupled expert router against the SAME eval_samples.json
used for the VLM baseline (rc_eval_unified.py), enabling direct comparison.

The router (frozen CLIP/BioMedCLIP + MLP) replaces VILA-M3 inference:
  image + question → backbone encoders → concat → MLP → {BRATS, VISTA3D, CXR}

Outputs rc_metrics.json and rc_metrics.csv in the identical format as the
VLM baseline, so numbers can be compared column-by-column.

Key difference vs. baseline:
  - Router always predicts one of 3 experts (no "none" class)
  - correct_route_rate = pure classification accuracy
  - RC measures consistency of routing across paraphrases of the same question

Usage:
  python eval_router.py \\
    --checkpoint  /path/router_out/router_biomedclip.pt \\
    --eval_samples /path/rc_eval_unified/eval_samples.json \\
    --out_dir     ./router_eval

If you have both clip and biomedclip checkpoints, run twice with different
--out_dir values and compare the resulting rc_metrics.json files.
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


IDX2EXPERT = {0: "BRATS", 1: "VISTA3D", 2: "CXR"}

DATASET_CONFIGS = {
    "brats":   {"categories": ["brats_grade", "brats_seg"], "expected_expert": "BRATS"},
    "vista3d": {"categories": ["vista3d_seg", "vista3d_what"], "expected_expert": "VISTA3D"},
    "cxr":     {"categories": ["cxr"],                        "expected_expert": "CXR"},
}


# ── MLP (must match train_router.py) ──────────────────────────────────────────

class RouterMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Backbone loading ───────────────────────────────────────────────────────────

def load_backbone(backbone_name: str):
    if backbone_name == "clip":
        from transformers import CLIPModel, CLIPProcessor
        print("Loading CLIP (openai/clip-vit-base-patch32)...")
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return ("clip", model, processor)

    if backbone_name == "biomedclip":
        import open_clip
        print("Loading BioMedCLIP...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return ("biomedclip", model, preprocess, tokenizer)

    raise ValueError(f"Unknown backbone: {backbone_name!r}")


def encode_sample(backbone_info: tuple, images: list, question: str, device: str) -> torch.Tensor:
    """Encode one sample → 1024-dim concat(img_feat, txt_feat) tensor."""
    name = backbone_info[0]

    if name == "clip":
        _, model, processor = backbone_info
        model.to(device)
        pil_imgs = [Image.open(p).convert("RGB") for p in images]
        img_in   = processor(images=pil_imgs, return_tensors="pt", padding=True)
        img_in   = {k: v.to(device) for k, v in img_in.items()}
        txt_in   = processor(text=[question], return_tensors="pt", padding=True,
                             truncation=True, max_length=77)
        txt_in   = {k: v.to(device) for k, v in txt_in.items()}
        with torch.no_grad():
            img_feat = model.get_image_features(**img_in)  # (N, 512)
            txt_feat = model.get_text_features(**txt_in)   # (1, 512)
        img_feat = nn.functional.normalize(img_feat, dim=-1).mean(dim=0, keepdim=True)  # (1, 512)
        txt_feat = nn.functional.normalize(txt_feat, dim=-1)
        return torch.cat([img_feat, txt_feat], dim=-1)  # (1, 1024)

    if name == "biomedclip":
        _, model, preprocess, tokenizer = backbone_info
        model.to(device)
        pil_imgs = [Image.open(p).convert("RGB") for p in images]
        tensors  = torch.stack([preprocess(img) for img in pil_imgs]).to(device)
        tokens   = tokenizer([question]).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(tensors)   # (N, 512)
            txt_feat = model.encode_text(tokens)     # (1, 512)
        img_feat = nn.functional.normalize(img_feat, dim=-1).mean(dim=0, keepdim=True)
        txt_feat = nn.functional.normalize(txt_feat, dim=-1)
        return torch.cat([img_feat, txt_feat], dim=-1)  # (1, 1024)

    raise ValueError(f"Unknown backbone: {name!r}")


# ── Metrics (mirrors rc_eval_unified.py exactly) ──────────────────────────────

def compute_rc(routings: list) -> float:
    n = len(routings)
    if n < 2:
        return 1.0
    agreements = sum(c * (c - 1) // 2 for c in Counter(routings).values())
    return agreements / (n * (n - 1) // 2)


def compute_global_re(routings: list) -> dict:
    n = len(routings)
    if n == 0:
        return {"H_norm": 0.0, "H_raw": 0.0, "K_experts": 0, "distribution": {}, "n_total": 0}
    counts = Counter(routings)
    active = [k for k in counts if k != "none"]
    K      = max(len(active), 2)
    h_raw  = -sum((c / n) * math.log(c / n) for c in counts.values())
    return {
        "H_norm":       round(h_raw / math.log(K), 6),
        "H_raw":        round(h_raw, 6),
        "K_experts":    K,
        "distribution": {k: round(v / n, 4) for k, v in counts.most_common()},
        "counts":       dict(counts),
        "n_total":      n,
    }


def compute_metrics_from_results(results: list, out_dir: str) -> dict:
    """Compute and write rc_metrics.json + rc_metrics.csv from result dicts."""
    print(f"\nComputing metrics from {len(results):,} results ...")

    # Per (dataset, item_id, category) → list of routing decisions
    groups: dict = defaultdict(list)
    for r in results:
        groups[(r["dataset"], r["item_id"], r["category"])].append(r["routing"])

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

    # Global Routing Entropy: average per-dataset p_k → equal dataset weighting
    per_ds_re = {
        ds: compute_global_re([r["routing"] for r in results if r["dataset"] == ds])
        for ds in DATASET_CONFIGS
    }

    all_experts = {"BRATS", "VISTA3D", "CXR", "none"}
    ds_list     = [ds for ds in DATASET_CONFIGS if per_ds_re[ds]["n_total"] > 0]
    avg_dist    = {
        exp: sum(per_ds_re[ds]["distribution"].get(exp, 0.0) for ds in ds_list) / len(ds_list)
        for exp in all_experts
    }
    total    = sum(avg_dist.values())
    avg_dist = {k: v / total for k, v in avg_dist.items() if v > 0}
    K        = len([k for k in avg_dist if k != "none"]) or 2
    h_raw    = -sum(p * math.log(p) for p in avg_dist.values())
    global_re = {
        "H_norm":       round(h_raw / math.log(K), 6),
        "H_raw":        round(h_raw, 6),
        "K_experts":    K,
        "distribution": {k: round(v, 4) for k, v in sorted(avg_dist.items(), key=lambda x: -x[1])},
        "note":         "averaged per-dataset p_k for equal dataset weighting",
    }

    metrics = {
        "global_routing_entropy":      global_re,
        "per_dataset_routing_entropy": per_ds_re,
        "per_dataset_category_rc":     {ds: dict(cats) for ds, cats in agg.items()},
        "per_item_category":           {
            f"{ds}__{item}__{cat}": v
            for (ds, item, cat), v in per_item_cat.items()
        },
    }

    # JSON
    json_path = os.path.join(out_dir, "rc_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON → {json_path}")

    # CSV
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

    # Console summary (same format as rc_eval_unified.py)
    print("\n" + "=" * 74)
    print("  Decoupled Router — Routing Consistency (RC)")
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
    print("  Routing Entropy (RE) — global  (H~1=adaptive, H~0=collapse)")
    print("=" * 74)
    g = global_re
    print(f"\n  Global  H_norm={g['H_norm']:.4f}  K={g['K_experts']}")
    print(f"  Distribution: { {k: f'{v:.3f}' for k,v in g['distribution'].items()} }")
    print(f"\n  Per-dataset RE:")
    for ds, re_d in per_ds_re.items():
        print(f"    [{ds.upper():<8s}]  H_norm={re_d['H_norm']:.4f}  dist={re_d['distribution']}")
    print("=" * 74)

    return metrics


# ── Inference ──────────────────────────────────────────────────────────────────

def run_router_inference(
    backbone_info: tuple,
    mlp: RouterMLP,
    samples: list,
    batch_size: int,
    device: str,
    checkpoint_path: str,
) -> list:
    """
    Run router on all eval samples. Supports resuming via JSONL checkpoint.
    Returns list of result dicts (same schema as rc_eval_unified.py inference_results.jsonl).
    """
    done_ids: set = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line.strip())["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"  Checkpoint: {len(done_ids):,} already done")

    pending = [s for s in samples if s["sample_id"] not in done_ids]
    print(f"  Pending:    {len(pending):,} samples")

    mlp.eval()
    results = []

    with open(checkpoint_path, "a") as ckpt_f:
        for i in tqdm(range(0, len(pending), batch_size), desc="Router inference"):
            batch = pending[i : i + batch_size]

            for s in batch:
                try:
                    img_paths = s["images"]
                    question  = s["question"]
                    feat      = encode_sample(backbone_info, img_paths, question, device)
                    with torch.no_grad():
                        logit   = mlp(feat.to(device))
                        pred_idx = logit.argmax(1).item()
                    routing = IDX2EXPERT[pred_idx]
                except Exception as e:
                    routing = "error"
                    print(f"  Error on {s['sample_id']}: {e}")

                rec = {
                    "sample_id": s["sample_id"],
                    "dataset":   s["dataset"],
                    "item_id":   s["item_id"],
                    "category":  s["category"],
                    "expected":  s["expected"],
                    "q_idx":     s["q_idx"],
                    "question":  s["question"],
                    "routing":   routing,
                }
                ckpt_f.write(json.dumps(rec) + "\n")
                ckpt_f.flush()
                results.append(rec)

    # Load all results (including previously done)
    all_results = []
    with open(checkpoint_path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if r.get("routing") not in ("error", None):
                    all_results.append(r)
            except json.JSONDecodeError:
                pass

    return all_results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint",   required=True,
                   help="Path to router_*.pt from train_router.py")
    p.add_argument("--eval_samples", required=True,
                   help="eval_samples.json produced by rc_eval_unified.py (same file as baseline)")
    p.add_argument("--out_dir",      default="./router_eval")
    p.add_argument("--batch_size",   type=int, default=1,
                   help="Inference batch size (1 is safest for mixed-image-count samples)")
    p.add_argument("--metrics_only", action="store_true",
                   help="Skip inference; compute metrics from existing checkpoint JSONL")
    p.add_argument("--datasets",     default="brats,vista3d,cxr",
                   help="Comma-separated subset of datasets to evaluate")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    datasets  = [d.strip() for d in args.datasets.split(",")]
    ckpt_path = os.path.join(out_dir, "router_inference_results.jsonl")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone_name = ckpt["backbone"]
    print(f"  backbone={backbone_name}  val_acc={ckpt.get('val_acc', '?'):.4f}  "
          f"epoch={ckpt.get('epoch', '?')}")

    mlp = RouterMLP(
        input_dim=ckpt["embed_dim"] * 2,
        hidden_dim=ckpt["hidden_dim"],
        num_classes=ckpt["num_classes"],
        dropout=ckpt.get("dropout", 0.1),
    )
    mlp.load_state_dict(ckpt["state_dict"])
    mlp.to(device)
    mlp.eval()

    # ── Load eval samples ──────────────────────────────────────────────────────
    print(f"Loading eval samples: {args.eval_samples}")
    with open(args.eval_samples) as f:
        all_samples = json.load(f)
    samples = [s for s in all_samples if s["dataset"] in datasets]
    print(f"  {len(samples):,} samples across datasets: {datasets}")

    # ── Inference ──────────────────────────────────────────────────────────────
    if not args.metrics_only:
        backbone_info = load_backbone(backbone_name)
        # Move backbone model to device
        backbone_model = backbone_info[1]
        backbone_model.to(device)

        results = run_router_inference(
            backbone_info, mlp, samples,
            batch_size=args.batch_size,
            device=device,
            checkpoint_path=ckpt_path,
        )
    else:
        print(f"Loading existing results from {ckpt_path}")
        results = []
        with open(ckpt_path) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    if r.get("routing") not in ("error", None) and r["dataset"] in datasets:
                        results.append(r)
                except json.JSONDecodeError:
                    pass
        print(f"  Loaded {len(results):,} valid results")

    # ── Metrics ────────────────────────────────────────────────────────────────
    compute_metrics_from_results(results, str(out_dir))


if __name__ == "__main__":
    main()
