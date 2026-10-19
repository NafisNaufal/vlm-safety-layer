#!/usr/bin/env python3
"""
train_router.py

Decoupled expert router for VILA-M3: frozen CLIP/BioMedCLIP image+text encoders
plus a trainable 2-layer MLP that classifies which expert to call.

Architecture (from paper diagram):
  image  → backbone image encoder (frozen) → f_img ∈ ℝ^512
  text   → backbone text encoder  (frozen) → f_txt ∈ ℝ^512
  concat → f ∈ ℝ^1024 → MLP(1024→512→3) → argmax → {BRATS, VISTA3D, CXR}

Labels: 0=BRATS, 1=VISTA3D, 2=CXR  (no "none" class — router always routes)

Training data comes from the train JSONs produced by prepare_training_data.py.
Only routing-relevant items are used (brats_what items are excluded since they
carry no expert token in the GPT response).

Features are pre-encoded once per backbone (GPU) and cached to .npz files, so
re-training the MLP is fast (~seconds per epoch on cached features).

Usage:
  python train_router.py \\
    --brats_train   /path/brats_train.json \\
    --vista3d_train /path/vista3d_train.json \\
    --cxr_train     /path/cxr_train.json \\
    --brats_val     /path/brats_val.json \\
    --vista3d_val   /path/vista3d_val.json \\
    --cxr_val       /path/cxr_val.json \\
    --out_dir       ./router_out \\
    --backbone      both

Produces:
  router_out/router_clip.pt         — CLIP checkpoint (if backbone=clip or both)
  router_out/router_biomedclip.pt   — BioMedCLIP checkpoint (if backbone=biomedclip or both)
  router_out/feats_train_*.npz      — cached encoded features
  router_out/feats_val_*.npz        — cached encoded features
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

LABEL_MAP  = {"brats": 0, "vista3d": 1, "cxr": 2}
IDX2EXPERT = {0: "BRATS", 1: "VISTA3D", 2: "CXR"}

# Separator used to split header from question in the human turn of each conversation
QUESTION_SEPS = {
    "brats":   "These are different MRI modalities.\n",
    "vista3d": "This is a CT image.\n",
    "cxr":     "This is a CXR image.\n",
}

# Routing token present in GPT turn 1 for routing conversations
ROUTING_TOKENS = {
    "brats":   "<BRATS()",
    "vista3d": "<VISTA3D(",
    "cxr":     "<CXR()>",
}


# ── Backbone wrappers (frozen) ─────────────────────────────────────────────────

class CLIPBackbone:
    name = "clip"
    embed_dim = 512

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        print("Loading CLIP (openai/clip-vit-base-patch32)...")
        self.model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def to(self, device):
        self.model.to(device)
        return self

    def encode_images(self, pil_images: list, device: str) -> torch.Tensor:
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return nn.functional.normalize(feats, dim=-1)  # (N, 512)

    def encode_texts(self, texts: list, device: str) -> torch.Tensor:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True,
            truncation=True, max_length=77,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return nn.functional.normalize(feats, dim=-1)  # (N, 512)


class BioMedCLIPBackbone:
    name = "biomedclip"
    embed_dim = 512

    def __init__(self):
        import open_clip
        print("Loading BioMedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def to(self, device):
        self.model.to(device)
        return self

    def encode_images(self, pil_images: list, device: str) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(device)
        with torch.no_grad():
            feats = self.model.encode_image(tensors)
        return nn.functional.normalize(feats, dim=-1)  # (N, 512)

    def encode_texts(self, texts: list, device: str) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        return nn.functional.normalize(feats, dim=-1)  # (N, 512)


def get_backbone(name: str):
    if name == "clip":
        return CLIPBackbone()
    if name == "biomedclip":
        return BioMedCLIPBackbone()
    raise ValueError(f"Unknown backbone: {name!r}. Use 'clip' or 'biomedclip'.")


# ── Data loading ───────────────────────────────────────────────────────────────

def extract_question(human_val: str, dataset: str) -> str:
    """Strip MODEL_CARDS + dataset header from the human turn, return bare question."""
    sep = QUESTION_SEPS.get(dataset)
    if sep and sep in human_val:
        return human_val.split(sep)[-1].strip()
    # BraTS what has no header — value IS the question (not used for training, but safe fallback)
    return human_val.replace("<image>", "").strip()


def load_routing_items(json_path: str, dataset: str, n_max: int, seed: int) -> list:
    """
    Load routing-relevant conversations from a train JSON.
    Filters out brats_what (no routing token in GPT response).
    Subsamples to n_max items.
    """
    print(f"  Loading {dataset} from {json_path} ...", flush=True)
    with open(json_path) as f:
        data = json.load(f)

    label     = LABEL_MAP[dataset]
    route_tok = ROUTING_TOKENS[dataset]
    items     = []

    for item in data:
        convs = item.get("conversations", [])
        if len(convs) < 2:
            continue
        if route_tok not in convs[1]["value"]:
            continue  # skip non-routing conversations (e.g. brats_what)

        question = extract_question(convs[0]["value"], dataset)
        if not question:
            continue

        img_paths = item.get("images", [])
        if dataset == "brats":
            img_paths = img_paths[:4]   # 4 MRI modalities; skip 5th (label overlay)
        else:
            img_paths = img_paths[:1]   # single image for VISTA3D and CXR

        if not img_paths:
            continue

        items.append({
            "images":   img_paths,
            "question": question,
            "label":    label,
        })

    if n_max > 0 and len(items) > n_max:
        items = random.Random(seed).sample(items, n_max)

    return items


# ── Feature encoding (with disk cache) ────────────────────────────────────────

def encode_dataset(
    backbone, items: list, batch_size: int, device: str, desc: str
) -> tuple:
    """
    Pre-encode all items → (img_feats, txt_feats, labels) as float32 numpy arrays.
    For multi-image items (BraTS 4-modality), image features are averaged.
    """
    all_img, all_txt, all_labels = [], [], []

    for i in tqdm(range(0, len(items), batch_size), desc=desc):
        batch = items[i : i + batch_size]

        # Image features: for BraTS, average across 4 modality images
        img_batch = []
        for it in batch:
            pil_imgs = [Image.open(p).convert("RGB") for p in it["images"]]
            feats = backbone.encode_images(pil_imgs, device)  # (n_imgs, 512)
            img_batch.append(feats.mean(dim=0))               # (512,)
        img_feats = torch.stack(img_batch)  # (B, 512)

        # Text features
        questions = [it["question"] for it in batch]
        txt_feats = backbone.encode_texts(questions, device)  # (B, 512)

        all_img.append(img_feats.cpu().float().numpy())
        all_txt.append(txt_feats.cpu().float().numpy())
        all_labels.extend(it["label"] for it in batch)

    return (
        np.concatenate(all_img, axis=0),
        np.concatenate(all_txt, axis=0),
        np.array(all_labels, dtype=np.int64),
    )


# ── MLP Router ─────────────────────────────────────────────────────────────────

class RouterMLP(nn.Module):
    def __init__(
        self, input_dim: int = 1024, hidden_dim: int = 512,
        num_classes: int = 3, dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, img_feats: np.ndarray, txt_feats: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(np.concatenate([img_feats, txt_feats], axis=1)).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_backbone(backbone_name: str, args, device: str) -> dict:
    """Train the router MLP for one backbone. Returns val accuracy and checkpoint path."""
    out_dir     = Path(args.out_dir)
    cache_train = out_dir / f"feats_train_{backbone_name}.npz"
    cache_val   = out_dir / f"feats_val_{backbone_name}.npz"

    print(f"\n{'=' * 62}")
    print(f"  Backbone: {backbone_name.upper()}")
    print(f"{'=' * 62}")

    # ── Load and subsample data ────────────────────────────────────────────────
    train_items, val_items = [], []
    for ds, tr_path, va_path in [
        ("brats",   args.brats_train,   args.brats_val),
        ("vista3d", args.vista3d_train, args.vista3d_val),
        ("cxr",     args.cxr_train,     args.cxr_val),
    ]:
        tr = load_routing_items(tr_path, ds, args.n_per_dataset, args.seed)
        va = load_routing_items(va_path, ds, args.n_val,         args.seed + 1)
        print(f"    {ds}: {len(tr):,} train  {len(va):,} val")
        train_items.extend(tr)
        val_items.extend(va)

    random.Random(args.seed).shuffle(train_items)
    random.Random(args.seed).shuffle(val_items)
    print(f"  Total: {len(train_items):,} train  {len(val_items):,} val")

    # ── Encode features (or load from cache) ──────────────────────────────────
    if cache_train.exists() and cache_val.exists() and not args.recompute_feats:
        print(f"  Loading cached features from {out_dir}")
        td = np.load(cache_train)
        vd = np.load(cache_val)
        tr_img, tr_txt, tr_lbl = td["img"], td["txt"], td["lbl"]
        va_img, va_txt, va_lbl = vd["img"], vd["txt"], vd["lbl"]
    else:
        backbone = get_backbone(backbone_name)
        backbone.to(device)
        tr_img, tr_txt, tr_lbl = encode_dataset(
            backbone, train_items, args.encode_batch, device,
            f"Encoding train ({backbone_name})",
        )
        va_img, va_txt, va_lbl = encode_dataset(
            backbone, val_items, args.encode_batch, device,
            f"Encoding val ({backbone_name})",
        )
        np.savez(cache_train, img=tr_img, txt=tr_txt, lbl=tr_lbl)
        np.savez(cache_val,   img=va_img, txt=va_txt, lbl=va_lbl)
        print(f"  Features cached → {cache_train}")
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Train MLP ──────────────────────────────────────────────────────────────
    train_loader = torch.utils.data.DataLoader(
        FeatureDataset(tr_img, tr_txt, tr_lbl),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        FeatureDataset(va_img, va_txt, va_lbl),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    embed_dim = 512  # both CLIP and BioMedCLIP output 512-dim features
    mlp = RouterMLP(
        input_dim=embed_dim * 2,
        hidden_dim=args.hidden_dim,
        num_classes=3,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    ckpt_path    = out_dir / f"router_{backbone_name}.pt"

    print(f"\n  Training MLP for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        mlp.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = mlp(X)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss    += loss.item() * len(y)
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_total   += len(y)
        scheduler.step()

        mlp.eval()
        val_correct, val_total = 0, 0
        per_cls_c = {0: 0, 1: 0, 2: 0}
        per_cls_t = {0: 0, 1: 0, 2: 0}
        with torch.no_grad():
            for X, y in val_loader:
                X, y   = X.to(device), y.to(device)
                preds  = mlp(X).argmax(1)
                val_correct += (preds == y).sum().item()
                val_total   += len(y)
                for p, g in zip(preds.cpu().tolist(), y.cpu().tolist()):
                    per_cls_t[g] += 1
                    if p == g:
                        per_cls_c[g] += 1

        val_acc   = val_correct / val_total
        train_acc = ep_correct  / ep_total
        cls_str   = "  ".join(
            f"{IDX2EXPERT[k]}={per_cls_c[k]/max(per_cls_t[k],1):.3f}"
            for k in range(3)
        )
        print(
            f"  Epoch {epoch:2d}/{args.epochs}"
            f"  loss={ep_loss/ep_total:.4f}"
            f"  train={train_acc:.4f}"
            f"  val={val_acc:.4f}"
            f"  [{cls_str}]"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "backbone":   backbone_name,
                    "embed_dim":  embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout":    args.dropout,
                    "num_classes": 3,
                    "state_dict": mlp.state_dict(),
                    "val_acc":    val_acc,
                    "epoch":      epoch,
                    "label_map":  LABEL_MAP,
                    "idx2expert": IDX2EXPERT,
                },
                ckpt_path,
            )

    print(f"\n  Best val_acc={best_val_acc:.4f}  saved → {ckpt_path}")
    return {"backbone": backbone_name, "val_acc": best_val_acc, "ckpt_path": str(ckpt_path)}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data
    p.add_argument("--brats_train",   required=True)
    p.add_argument("--vista3d_train", required=True)
    p.add_argument("--cxr_train",     required=True)
    p.add_argument("--brats_val",     required=True)
    p.add_argument("--vista3d_val",   required=True)
    p.add_argument("--cxr_val",       required=True)

    # Backbone
    p.add_argument("--backbone", default="both",
                   choices=["clip", "biomedclip", "both"],
                   help="Which encoder to use (both = train clip + biomedclip, pick best)")

    # Sampling
    p.add_argument("--n_per_dataset", type=int, default=20_000,
                   help="Max train items per dataset (0 = use all)")
    p.add_argument("--n_val",         type=int, default=5_000,
                   help="Max val items per dataset (0 = use all)")
    p.add_argument("--seed",          type=int, default=42)

    # MLP
    p.add_argument("--hidden_dim", type=int,   default=512)
    p.add_argument("--dropout",    type=float, default=0.1)

    # Training
    p.add_argument("--epochs",       type=int,   default=15)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--encode_batch", type=int,   default=32,
                   help="Batch size for feature encoding (GPU memory limited)")

    # Output
    p.add_argument("--out_dir",         default="./router_out")
    p.add_argument("--recompute_feats", action="store_true",
                   help="Re-encode features even if .npz cache exists")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    backbones = ["clip", "biomedclip"] if args.backbone == "both" else [args.backbone]

    results = []
    for bb in backbones:
        res = train_one_backbone(bb, args, device)
        results.append(res)

    if len(results) > 1:
        print("\n" + "=" * 62)
        print("  Final comparison")
        print("=" * 62)
        for r in sorted(results, key=lambda x: -x["val_acc"]):
            print(f"  {r['backbone']:<12s}  val_acc={r['val_acc']:.4f}  ckpt={r['ckpt_path']}")
        winner = max(results, key=lambda x: x["val_acc"])
        print(f"\n  Recommended for eval: --backbone {winner['backbone']} "
              f"--checkpoint {winner['ckpt_path']}")


if __name__ == "__main__":
    main()
