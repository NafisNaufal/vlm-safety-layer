#!/usr/bin/env python3
"""
rc_eval_pipeline.py

End-to-end Routing Consistency (RC) evaluation pipeline for VILA-M3 on BraTS 2018.

Pipeline stages:
  1. Paraphrase generation   — google/gemma-4-E4B-it generates 10 semantically
                               equivalent variants per seed (grade + segmentation).
                               Description category uses 20 existing LLaMA questions.
  2. Eval sample generation  — Deduplicates slices across train/val/test conversation
                               JSONs (~4.8k unique slices) and builds one VILA-M3
                               conversation per (slice x question variant).
  3. VILA-M3 inference       — MONAI/Llama3-VILA-M3-8B processes each eval sample
                               (greedy, max 64 tokens). Results are appended to a
                               JSONL checkpoint so the job can be interrupted/resumed.
  4. RC metric computation   — Aggregates routing decisions per (slice, category),
                               computes RC scores, writes JSON + CSV reports.

Routing Consistency formula:
  RC(group) = sum_k C(count_k, 2) / C(n, 2)
  count_k = # queries routed to expert k, n = total queries in group.
  RC=1.0 means all variants route identically; RC=0.0 means all differ.

Outputs (all under --out-dir, default ~/projects/BraTS/rc_eval):
  rc_paraphrase_groups.json    paraphrase group metadata (skipped if exists)
  rc_eval_samples.json         all (slice x question) eval sample dicts (skipped if exists)
  rc_inference_results.jsonl   one JSON line per result (append + resume safe)
  rc_results.json              full RC metric report
  rc_results.csv               flat CSV: slice_id, category, rc_score, ...

Estimated runtime on single A100 (80GB):
  Phase 1 (Gemma paraphrases): ~2 min
  Phase 2 (build samples):     ~5 min
  Phase 3 (VILA-M3 inference): ~200k samples x ~0.6s = ~33 h  (resumable overnight job)
  Phase 4 (metrics):           ~1 min

Usage:
  python rc_eval_pipeline.py \\
    --gemma-model google/gemma-4-E4B-it \\
    --vila-model MONAI/Llama3-VILA-M3-8B \\
    --conversations-dir ~/projects/BraTS/vila_m3_conversations \\
    --slices-dir ~/projects/BraTS/vila_m3_slices \\
    --remote-prefix /path/to/BraTS/vila_m3_slices \\
    --llama-questions /path/to/VLM-Radiology-Agent-Framework/m3/data_prepare/experts/llama_output/llama_gen_expert_what.txt \\
    --out-dir ~/projects/BraTS/rc_eval

Resume after interruption (inference checkpoint is preserved):
  python rc_eval_pipeline.py [same args]   # automatically skips done samples

Compute metrics from a completed checkpoint without re-running inference:
  python rc_eval_pipeline.py --metrics-only [same args]
"""

import argparse
import csv
import gc
import json
import math
import os
import re
from collections import Counter, defaultdict

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMMA_CATEGORIES = {
    "segmentation": {
        "seed": "Segment any tumors in the image",
        "expected_expert": "BRATS",
        "context": (
            "brain MRI tumor segmentation — the viewer wants the tumors "
            "delineated/outlined/segmented in the brain MRI scan"
        ),
    },
    "grade": {
        "seed": "What glioma grade is shown in the image?",
        "expected_expert": "BRATS",
        "context": (
            "brain tumor grade classification from MRI — the viewer wants to know "
            "whether the tumor is high-grade (HGG) or low-grade (LGG) glioma"
        ),
    },
}

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

PARAPHRASE_PROMPT = """\
You are a medical imaging assistant helping evaluate a VLM routing system for brain MRI analysis.

Your task: generate {n} semantically equivalent paraphrases of the question below.

Rules:
- Every paraphrase must express the exact same intent as the original
- Vary vocabulary, sentence structure, and phrasing
- Maintain a clinical/radiological register (as a radiologist or clinician would phrase it)
- Do NOT add new conditions, constraints, or change what is being asked
- Context: {context}

Original question: "{seed}"

Output exactly {n} numbered paraphrases, one per line (format: "1. <question>").
Do not include any explanations or extra text.
"""

# Replicates the demo's mentioned_by() check exactly:
#   re.findall(r"<(.*?)>", output) must return exactly 1 match,
#   and that match must contain the expert name.
_ALL_ANGLE_RE = re.compile(r"<(.*?)>")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def remap_path(path: str, remote_prefix: str, local_dir: str) -> str:
    """Replace the remote NAS prefix with the local slices directory."""
    remote_prefix = remote_prefix.rstrip("/") + "/"
    local_dir = os.path.expanduser(local_dir).rstrip("/") + "/"
    if path.startswith(remote_prefix):
        return local_dir + path[len(remote_prefix):]
    if not os.path.isabs(path):
        return local_dir + path
    return path


def _parse_numbered_list(text: str, expected: int) -> list:
    """Extract items from a numbered list ('1.', '1)', '1:' prefixes)."""
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[\.\)\:]?\s+(.+)$", line)
        if m:
            results.append(m.group(1).strip())
    return results[:expected]


def parse_routing(text: str) -> str:
    """
    Replicate the VILA-M3 demo's mentioned_by() check exactly.
    The demo does: re.findall(r"<(.*?)>", output) and requires EXACTLY 1 match.
    Zero matches → 'none'. Two or more matches → 'none' (would not trigger any expert).
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


def compute_rc_score(routing_decisions: list) -> float:
    """
    Fraction of question-pair comparisons that agree on routing.
    RC = sum_k C(count_k, 2) / C(n, 2)
    """
    n = len(routing_decisions)
    if n < 2:
        return 1.0
    pair_agreements = sum(c * (c - 1) // 2 for c in Counter(routing_decisions).values())
    total_pairs = n * (n - 1) // 2
    return pair_agreements / total_pairs


def compute_routing_entropy(routing_decisions: list, n_experts: int = 4) -> float:
    """
    Normalized Shannon entropy of the routing distribution.
    H = -Σ p_k log2(p_k) / log2(n_experts)
    0.0 = all questions routed identically, 1.0 = maximally spread across experts.
    n_experts=4 covers {BRATS, VISTA3D, CXR, none}.
    """
    n = len(routing_decisions)
    if n == 0:
        return 0.0
    counts = Counter(routing_decisions)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    return h / math.log2(n_experts)


# ---------------------------------------------------------------------------
# Phase 1 — Gemma paraphrase generation
# ---------------------------------------------------------------------------

def load_gemma(model_id: str, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading Gemma tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"Loading Gemma model: {model_id}  dtype={dtype}  device_map=auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print("Gemma loaded.")
    return model, tokenizer


def generate_paraphrases(
    model,
    tokenizer,
    seed: str,
    context: str,
    n: int,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
) -> list:
    prompt = PARAPHRASE_PROMPT.format(n=n, context=context, seed=seed)
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = (
            f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    paraphrases = _parse_numbered_list(raw_output, expected=n)
    if len(paraphrases) < n:
        print(
            f"  Warning: requested {n} paraphrases, parsed {len(paraphrases)}.\n"
            f"  Raw output:\n{raw_output}\n"
        )
    return paraphrases


def build_paraphrase_groups(
    model, tokenizer, llama_questions_path: str, n_paraphrases: int
) -> list:
    groups = []

    # Description — existing LLaMA questions used directly (no Gemma needed)
    path = os.path.expanduser(llama_questions_path)
    print(f"\nLoading LLaMA description questions: {path}")
    with open(path) as f:
        llama_qs = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(llama_qs)} questions")
    groups.append({
        "category": "description",
        "expected_expert": "none",
        "source": "llama_generated",
        "seed": None,
        "paraphrases": llama_qs,
        "all_queries": llama_qs,
        "n_queries": len(llama_qs),
        "n_pairs": len(llama_qs) * (len(llama_qs) - 1) // 2,
    })

    # Grade + Segmentation — generate with Gemma
    for category, cfg in GEMMA_CATEGORIES.items():
        seed, context, expected_expert = cfg["seed"], cfg["context"], cfg["expected_expert"]
        print(f"\n[{category}] Generating {n_paraphrases} paraphrases for: '{seed}'")
        paraphrases = generate_paraphrases(model, tokenizer, seed, context, n=n_paraphrases)
        print(f"  Parsed {len(paraphrases)} paraphrases:")
        for i, p in enumerate(paraphrases, 1):
            print(f"    {i:2d}. {p}")
        all_queries = [seed] + paraphrases
        groups.append({
            "category": category,
            "expected_expert": expected_expert,
            "source": "gemma_generated",
            "seed": seed,
            "paraphrases": paraphrases,
            "all_queries": all_queries,
            "n_queries": len(all_queries),
            "n_pairs": len(all_queries) * (len(all_queries) - 1) // 2,
        })

    return groups


def unload_gemma(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Gemma unloaded from GPU.")


# ---------------------------------------------------------------------------
# Phase 2 — Eval sample generation
# ---------------------------------------------------------------------------

def load_unique_slices(
    conversations_dir: str, remote_prefix: str, slices_dir: str
) -> list:
    """
    Read all three conversation JSONs, deduplicate by the 4-modality image key,
    and return slices where all 4 local PNG files exist.
    """
    conversations_dir = os.path.expanduser(conversations_dir)
    seen = {}  # tuple(4 local paths) -> slice dict

    for split in ("train", "val", "test"):
        jf = os.path.join(conversations_dir, f"brats2018_{split}.json")
        if not os.path.exists(jf):
            print(f"  Warning: {jf} not found, skipping")
            continue
        with open(jf) as f:
            data = json.load(f)
        print(f"  {split}: {len(data)} entries")

        for item in data:
            raw_images = item.get("images", [])
            if len(raw_images) < 4:
                continue
            local_images = [
                remap_path(p, remote_prefix, slices_dir) for p in raw_images[:4]
            ]
            key = tuple(local_images)
            if key in seen:
                continue
            t1ce = local_images[0]
            subject = os.path.basename(os.path.dirname(t1ce))
            base = os.path.basename(t1ce).replace("_img_t1ce.png", "")
            seen[key] = {
                "slice_id": f"{subject}/{base}",
                "images": list(local_images),
            }

    valid, missing = [], 0
    for s in seen.values():
        if all(os.path.exists(p) for p in s["images"]):
            valid.append(s)
        else:
            missing += 1
    if missing:
        print(f"  Warning: {missing} slices skipped (missing local PNG files)")
    print(f"  Unique valid slices: {len(valid)}")
    return valid


def build_eval_conversation(question: str, images: list) -> dict:
    human_value = (
        MODEL_CARDS
        + "T1(contrast enhanced): <image>, T1: <image>, T2: <image>, FLAIR: <image> "
        "These are different MRI modalities.\n"
        + question
    )
    return {
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": ""},
        ],
        "images": images,
    }


def generate_eval_samples(slices: list, paraphrase_groups: list) -> list:
    samples = []
    for slice_info in tqdm(slices, desc="Building eval samples"):
        for group in paraphrase_groups:
            for q_idx, question in enumerate(group["all_queries"]):
                sample = build_eval_conversation(question, slice_info["images"])
                sample.update({
                    "sample_id": (
                        f"{slice_info['slice_id']}__{group['category']}__{q_idx:02d}"
                    ),
                    "slice_id": slice_info["slice_id"],
                    "category": group["category"],
                    "expected_expert": group["expected_expert"],
                    "group_id": group["category"],
                    "is_seed": (
                        q_idx == 0 and group.get("source") == "gemma_generated"
                    ),
                    "question": question,
                    "q_idx": q_idx,
                })
                samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Phase 3 — VILA-M3 inference
# ---------------------------------------------------------------------------

def _mock_deepspeed():
    """Prevent deepspeed training imports from crashing llava on inference-only machines."""
    import sys
    import types
    import importlib.machinery

    if "deepspeed" in sys.modules:
        return

    ds = types.ModuleType("deepspeed")
    ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    ds.__path__ = []  # makes it look like a package to find_spec

    ds_comm = types.ModuleType("deepspeed.comm")
    ds_comm.__spec__ = importlib.machinery.ModuleSpec("deepspeed.comm", None)
    for fn in ("is_initialized", "barrier", "all_reduce", "broadcast", "new_group"):
        setattr(ds_comm, fn, lambda *a, **kw: None)
    ds_comm.get_rank = lambda: 0
    ds_comm.get_world_size = lambda: 1

    sys.modules["deepspeed"] = ds
    sys.modules["deepspeed.comm"] = ds_comm


def load_vila(model_path: str, model_base, conv_mode: str):
    """Load VILA-M3. Downloads from HuggingFace if model_path is not a local directory."""
    _mock_deepspeed()

    # Newer transformers removed is_tf_available from transformers.utils; llava still imports it
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
    print(f"  Model name: {model_name}  conv_mode: {conv_mode}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        resolved, model_name, model_base
    )
    print(f"  Context length: {context_len}")
    cuda_ok = torch.cuda.is_available() and next(model.parameters()).is_cuda
    print(f"  cuda: {cuda_ok}")
    if not cuda_ok:
        print("  WARNING: no GPU — converting model to float32 for CPU inference")
        model = model.float()
    return tokenizer, model, image_processor


def run_inference(
    tokenizer,
    model,
    image_processor,
    conv_mode: str,
    samples: list,
    checkpoint_path: str,
    max_new_tokens: int = 64,
):
    """
    Run VILA-M3 inference on all samples with per-line JSONL checkpointing.
    Already-completed sample_ids are skipped so the job can be safely resumed.
    """
    _mock_deepspeed()

    import transformers.utils as _tutils
    if not hasattr(_tutils, "is_tf_available"):
        _tutils.is_tf_available = lambda: False

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
    from PIL import Image

    # Load completed sample ids from checkpoint
    done_ids: set = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"  Checkpoint: {len(done_ids):,} samples already done")

    pending = [s for s in samples if s["sample_id"] not in done_ids]
    print(f"  Remaining : {len(pending):,} samples")

    if not pending:
        print("  Nothing to do — all samples already in checkpoint.")
        return

    conv_template = conv_templates[conv_mode]
    raw_stop = (
        conv_template.sep
        if conv_template.sep_style != SeparatorStyle.TWO
        else conv_template.sep2
    )
    # llama_3 uses "" as sep; fall back to <|eot_id|> so we don't stop immediately
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
                    tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .to(model.device)
                )
                stopping_criteria = KeywordsStoppingCriteria(
                    [stop_str], tokenizer, input_ids
                )

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=[images_tensor],
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # VILA's generate returns only new tokens (not full input+output)
                if output_ids.shape[1] <= input_ids.shape[1]:
                    new_ids = output_ids[0]
                else:
                    new_ids = output_ids[0][input_ids.shape[1]:]
                output_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                if output_text.endswith(stop_str):
                    output_text = output_text[: -len(stop_str)].strip()
                routing = parse_routing(output_text)
                result = {
                    "sample_id": sample["sample_id"],
                    "slice_id": sample["slice_id"],
                    "category": sample["category"],
                    "expected_expert": sample["expected_expert"],
                    "q_idx": sample["q_idx"],
                    "question": sample["question"],
                    "output": output_text,
                    "routing": routing,
                }

            except Exception as exc:
                result = {
                    "sample_id": sample["sample_id"],
                    "slice_id": sample["slice_id"],
                    "category": sample["category"],
                    "expected_expert": sample["expected_expert"],
                    "q_idx": sample["q_idx"],
                    "question": sample["question"],
                    "output": f"ERROR: {exc}",
                    "routing": "error",
                }

            ckpt_f.write(json.dumps(result) + "\n")
            ckpt_f.flush()


# ---------------------------------------------------------------------------
# Phase 4 — RC metric computation
# ---------------------------------------------------------------------------

def compute_metrics(checkpoint_path: str, paraphrase_groups: list, out_dir: str) -> dict:
    # Load valid results (skip error entries)
    results = []
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("routing") not in ("error", None):
                    # Re-parse routing from output to apply any regex fixes
                    if "output" in r:
                        r["routing"] = parse_routing(r["output"])
                    results.append(r)
            except json.JSONDecodeError:
                pass
    print(f"\nLoaded {len(results):,} valid inference results")

    # Group by (slice_id, category)
    groups: dict = defaultdict(list)
    for r in results:
        groups[(r["slice_id"], r["category"])].append(r["routing"])

    # Per-(slice, category) RC + entropy
    per_slice_cat = {}
    for (slice_id, category), routings in groups.items():
        counts = Counter(routings)
        per_slice_cat[(slice_id, category)] = {
            "rc_score": compute_rc_score(routings),
            "routing_entropy": compute_routing_entropy(routings),
            "n_queries": len(routings),
            "routing_distribution": dict(counts),
            "majority_routing": counts.most_common(1)[0][0],
        }

    # Per-category aggregate
    all_categories = [g["category"] for g in paraphrase_groups]
    category_stats = {}
    for cat in all_categories:
        rc_scores  = [v["rc_score"]        for (s, c), v in per_slice_cat.items() if c == cat]
        ent_scores = [v["routing_entropy"]  for (s, c), v in per_slice_cat.items() if c == cat]
        majority_counts: dict = defaultdict(int)
        for (s, c), v in per_slice_cat.items():
            if c == cat:
                majority_counts[v["majority_routing"]] += 1
        if rc_scores:
            sorted_rc  = sorted(rc_scores)
            sorted_ent = sorted(ent_scores)
            median_rc  = sorted_rc[len(sorted_rc) // 2]
            median_ent = sorted_ent[len(sorted_ent) // 2]
        else:
            median_rc = median_ent = 0.0
        category_stats[cat] = {
            "mean_rc":      sum(rc_scores)  / len(rc_scores)  if rc_scores  else 0.0,
            "median_rc":    median_rc,
            "mean_entropy": sum(ent_scores) / len(ent_scores) if ent_scores else 0.0,
            "median_entropy": median_ent,
            "n_slices": len(rc_scores),
            "majority_routing_counts": dict(majority_counts),
        }

    overall_rc_all = [v["rc_score"] for v in per_slice_cat.values()]
    overall_rc = sum(overall_rc_all) / len(overall_rc_all) if overall_rc_all else 0.0

    metrics = {
        "overall_rc": overall_rc,
        "n_slice_category_groups": len(per_slice_cat),
        "per_category": category_stats,
        "per_slice_category": {
            f"{s}__{c}": v for (s, c), v in per_slice_cat.items()
        },
    }

    # JSON
    json_path = os.path.join(out_dir, "rc_results.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"RC results JSON → {json_path}")

    # CSV
    csv_path = os.path.join(out_dir, "rc_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "slice_id", "category", "rc_score", "routing_entropy", "n_queries",
            "majority_routing", "routing_distribution",
        ])
        for (slice_id, cat), v in sorted(per_slice_cat.items()):
            writer.writerow([
                slice_id, cat,
                f"{v['rc_score']:.6f}",
                f"{v['routing_entropy']:.6f}",
                v["n_queries"],
                v["majority_routing"],
                json.dumps(v["routing_distribution"]),
            ])
    print(f"RC results CSV  → {csv_path}")

    # Console summary
    print("\n" + "=" * 62)
    print("  Routing Consistency — Summary")
    print("=" * 62)
    print(
        f"  Overall mean RC : {overall_rc:.4f}"
        f"  ({len(per_slice_cat):,} slice x category groups)"
    )
    print()
    for cat in all_categories:
        s = category_stats[cat]
        routing_str = "  ".join(
            f"{k}={v}"
            for k, v in sorted(
                s["majority_routing_counts"].items(), key=lambda x: -x[1]
            )
        )
        print(
            f"  [{cat:12s}]  RC={s['mean_rc']:.4f} (median {s['median_rc']:.4f})"
            f"  H={s['mean_entropy']:.4f} (median {s['median_entropy']:.4f})"
            f"  n={s['n_slices']:,}"
        )
        print(f"               majority routing: {routing_str}")
    print("=" * 62)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="VILA-M3 Routing Consistency evaluation pipeline — BraTS 2018"
    )
    p.add_argument(
        "--gemma-model", default="google/gemma-4-E4B-it",
        help="HuggingFace repo or local path for paraphrase generation model",
    )
    p.add_argument(
        "--vila-model", default="MONAI/Llama3-VILA-M3-8B",
        help="HuggingFace repo or local path for VILA-M3 checkpoint",
    )
    p.add_argument(
        "--model-base", default=None,
        help="Base model path for LoRA VILA-M3 (leave None for full checkpoints)",
    )
    p.add_argument(
        "--conv-mode", default="llama_3",
        help="VILA-M3 conversation mode (default: llama_3)",
    )
    p.add_argument(
        "--conversations-dir",
        default="~/projects/BraTS/vila_m3_conversations",
        help="Directory containing brats2018_{train,val,test}.json",
    )
    p.add_argument(
        "--slices-dir",
        default="~/projects/BraTS/vila_m3_slices",
        help="Local directory containing the PNG slice files",
    )
    p.add_argument(
        "--remote-prefix",
        default="/path/to/BraTS/vila_m3_slices",
        help="Remote NAS prefix in the JSON image paths (replaced by --slices-dir)",
    )
    p.add_argument(
        "--llama-questions",
        default=(
            "/path/to/VLM-Radiology-Agent-Framework"
            "/m3/data_prepare/experts/llama_output/llama_gen_expert_what.txt"
        ),
        help="Path to LLaMA-generated description questions",
    )
    p.add_argument(
        "--out-dir", default="~/projects/BraTS/rc_eval",
        help="Output directory for all generated files",
    )
    p.add_argument(
        "--n-paraphrases", type=int, default=10,
        help="Number of Gemma paraphrases per seed question (grade + segmentation)",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Max tokens for VILA-M3 routing decision (64 is sufficient)",
    )
    p.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16"],
        help="Gemma dtype (bfloat16 recommended for A100)",
    )
    p.add_argument(
        "--limit-slices", type=int, default=-1,
        help="Cap number of unique slices for quick testing (-1 = no limit)",
    )
    p.add_argument(
        "--force-regen-paraphrases", action="store_true",
        help="Regenerate paraphrase groups even if rc_paraphrase_groups.json exists",
    )
    p.add_argument(
        "--force-regen-samples", action="store_true",
        help="Regenerate eval samples even if rc_eval_samples.json exists",
    )
    p.add_argument(
        "--categories", default="grade,segmentation",
        help="Comma-separated categories to evaluate (default: grade,segmentation)",
    )
    p.add_argument(
        "--metrics-only", action="store_true",
        help=(
            "Skip paraphrase gen and inference; compute metrics from "
            "existing rc_inference_results.jsonl + rc_paraphrase_groups.json"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    groups_path = os.path.join(out_dir, "rc_paraphrase_groups.json")
    samples_path = os.path.join(out_dir, "rc_eval_samples.json")
    checkpoint_path = os.path.join(out_dir, "rc_inference_results.jsonl")

    # ── Phase 1: Paraphrase generation ───────────────────────────────────────
    if args.metrics_only:
        print(f"--metrics-only: loading paraphrase groups from {groups_path}")
        with open(groups_path) as f:
            paraphrase_groups = json.load(f)

    elif os.path.exists(groups_path) and not args.force_regen_paraphrases:
        print(f"Phase 1 skipped — paraphrase groups already exist: {groups_path}")
        with open(groups_path) as f:
            paraphrase_groups = json.load(f)

    else:
        print("\n── Phase 1: Generating paraphrases with Gemma ──────────────────")
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        gemma, g_tok = load_gemma(args.gemma_model, dtype=dtype)
        paraphrase_groups = build_paraphrase_groups(
            gemma, g_tok,
            llama_questions_path=args.llama_questions,
            n_paraphrases=args.n_paraphrases,
        )
        unload_gemma(gemma, g_tok)

        with open(groups_path, "w") as f:
            json.dump(paraphrase_groups, f, indent=2)
        print(f"\nParaphrase groups → {groups_path}")

    total_queries = sum(g["n_queries"] for g in paraphrase_groups)
    print(
        f"\nParaphrase groups: {len(paraphrase_groups)} categories, "
        f"{total_queries} total queries per slice"
    )
    for g in paraphrase_groups:
        print(f"  [{g['category']:12s}]  {g['n_queries']} queries  ({g['n_pairs']} pairs)")

    # ── Phase 2: Eval sample generation ──────────────────────────────────────
    if args.metrics_only:
        print(f"\nPhase 2 skipped — loading eval samples from {samples_path}")
        with open(samples_path) as f:
            eval_samples = json.load(f)

    elif os.path.exists(samples_path) and not args.force_regen_samples:
        print(f"\nPhase 2 skipped — eval samples already exist: {samples_path}")
        with open(samples_path) as f:
            eval_samples = json.load(f)

    else:
        print("\n── Phase 2: Building eval samples ──────────────────────────────")
        print("\nLoading unique slices from train + val + test ...")
        unique_slices = load_unique_slices(
            conversations_dir=args.conversations_dir,
            remote_prefix=args.remote_prefix,
            slices_dir=args.slices_dir,
        )
        if args.limit_slices > 0:
            print(f"  Capping at {args.limit_slices} slices (--limit-slices)")
            unique_slices = unique_slices[: args.limit_slices]

        eval_samples = generate_eval_samples(unique_slices, paraphrase_groups)

        with open(samples_path, "w") as f:
            json.dump(eval_samples, f)
        print(f"\nEval samples ({len(eval_samples):,}) → {samples_path}")

    print(f"\nTotal eval samples (all categories): {len(eval_samples):,}")

    # Filter to requested categories before inference + metrics
    active_cats = set(args.categories.split(","))
    eval_samples = [s for s in eval_samples if s["category"] in active_cats]
    paraphrase_groups = [g for g in paraphrase_groups if g["category"] in active_cats]
    print(f"Active categories: {sorted(active_cats)}  →  {len(eval_samples):,} samples")

    # ── Phase 3: VILA-M3 inference ────────────────────────────────────────────
    if not args.metrics_only:
        print("\n── Phase 3: VILA-M3 inference ──────────────────────────────────")
        vila_tok, vila_model, vila_proc = load_vila(
            model_path=args.vila_model,
            model_base=args.model_base,
            conv_mode=args.conv_mode,
        )
        run_inference(
            tokenizer=vila_tok,
            model=vila_model,
            image_processor=vila_proc,
            conv_mode=args.conv_mode,
            samples=eval_samples,
            checkpoint_path=checkpoint_path,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"\nInference checkpoint → {checkpoint_path}")

    # ── Phase 4: RC metrics ───────────────────────────────────────────────────
    print("\n── Phase 4: Computing RC metrics ───────────────────────────────")
    if not os.path.exists(checkpoint_path):
        print("No inference results found. Run without --metrics-only first.")
        return

    compute_metrics(checkpoint_path, paraphrase_groups, out_dir)


if __name__ == "__main__":
    main()
