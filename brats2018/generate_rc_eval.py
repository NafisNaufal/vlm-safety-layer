#!/usr/bin/env python3
"""
generate_rc_eval.py

Generate semantically similar VQA paraphrase groups for Routing Consistency (RC) evaluation.

Description category: uses the 20 existing LLaMA-generated questions from the repo
                      (m3/data_prepare/experts/llama_output/llama_gen_expert_what.txt)
Grade + Segmentation: uses google/gemma-4-26B-A4B-it to generate N paraphrases

Outputs:
  <out_dir>/rc_paraphrase_groups.json   -- paraphrase groups (all queries per category)
  <out_dir>/rc_eval_samples.json        -- eval-ready samples in VILA-M3 conversation format

Usage:
  python generate_rc_eval.py \
    --model google/gemma-4-26B-A4B-it \
    --test-json vila_m3_conversations/brats2018_test.json \
    --slices-dir vila_m3_slices \
    --llama-questions VLM-Radiology-Agent-Framework/m3/data_prepare/experts/llama_output/llama_gen_expert_what.txt \
    --out-dir vila_m3_conversations \
    --n-paraphrases 10
"""

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Seed questions — categories that need Gemma paraphrases
# Description is excluded: its paraphrase group comes from the LLaMA file directly.
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

# Exact MODEL_CARDS string used by VILA-M3 (from gradio_m3.py)
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, dtype=torch.bfloat16):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model: {model_id} (dtype={dtype}, device_map=auto)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Paraphrase generation
# ---------------------------------------------------------------------------

def generate_paraphrases(
    model,
    tokenizer,
    seed: str,
    context: str,
    n: int,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
) -> list[str]:
    prompt = PARAPHRASE_PROMPT.format(n=n, context=context, seed=seed)

    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback if chat template not available
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

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

    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    paraphrases = _parse_numbered_list(raw_output, expected=n)

    if len(paraphrases) < n:
        print(
            f"  Warning: requested {n} paraphrases but parsed {len(paraphrases)}. "
            f"Raw output below:\n{raw_output}\n"
        )

    return paraphrases


def _parse_numbered_list(text: str, expected: int) -> list[str]:
    """Extract items from a numbered list (handles '1.', '1)', '1:' prefixes)."""
    lines = text.strip().split("\n")
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+[\.\)\:]?\s+(.+)$", line)
        if match:
            results.append(match.group(1).strip())
    return results[:expected]


# ---------------------------------------------------------------------------
# Fixed image selection
# ---------------------------------------------------------------------------

def remap_path(path: str, remote_prefix: str, local_slices_dir: str) -> str:
    """Replace the remote NAS prefix with the local slices directory."""
    # Normalize both
    remote_prefix = remote_prefix.rstrip("/") + "/"
    local_slices_dir = local_slices_dir.rstrip("/") + "/"

    if path.startswith(remote_prefix):
        return local_slices_dir + path[len(remote_prefix):]
    # If path is already relative (just subject/file), prepend local dir
    if not os.path.isabs(path):
        return local_slices_dir + path
    return path


def find_fixed_image(test_json_path: str, local_slices_dir: str, remote_prefix: str) -> list[str]:
    """
    Pick one representative test slice (4 MRI modality images, no label).
    Prefers a slice from a grade-classification conversation (confirms tumor present).
    Falls back to any available slice.
    """
    with open(test_json_path) as f:
        test_data = json.load(f)

    for item in test_data:
        # Only look at 4-image items (grade and segmentation samples have 5 images: 4 + label)
        # We want the 4 modality images only (no label/seg image for the eval prompt)
        raw_images = item.get("images", [])
        if len(raw_images) < 4:
            continue

        # Use only the first 4 images (T1ce, T1, T2, FLAIR) — drop the label
        local_images = [
            remap_path(p, remote_prefix, local_slices_dir)
            for p in raw_images[:4]
        ]

        if all(os.path.exists(p) for p in local_images):
            print(f"Fixed image: {os.path.dirname(local_images[0])}")
            return local_images

    raise RuntimeError(
        "Could not find a valid test slice with all 4 modality images present locally."
    )


# ---------------------------------------------------------------------------
# Build eval conversation (VILA-M3 format)
# ---------------------------------------------------------------------------

def build_eval_conversation(question: str, images: list[str]) -> dict:
    """
    Single-turn conversation for RC evaluation.
    Matches the format VILA-M3 expects during inference:
      MODEL_CARDS + image tokens + modality note + question
    The GPT response is left empty — to be filled by VILA-M3.
    """
    human_value = (
        MODEL_CARDS
        + "T1(contrast enhanced): <image>, T1: <image>, T2: <image>, FLAIR: <image> "
        "These are different MRI modalities.\n"
        + question
    )
    return {
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": ""},  # filled by VILA-M3 at inference
        ],
        "images": images,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate RC evaluation paraphrases with Gemma")
    p.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    p.add_argument(
        "--test-json",
        default="vila_m3_conversations/brats2018_test.json",
        help="Path to brats2018_test.json",
    )
    p.add_argument(
        "--slices-dir",
        default="vila_m3_slices",
        help="Local path to the extracted slice PNG directory",
    )
    p.add_argument(
        "--remote-prefix",
        default="/path/to/BraTS/vila_m3_slices",
        help="Remote NAS prefix used in the JSON image paths (will be replaced by --slices-dir)",
    )
    p.add_argument(
        "--llama-questions",
        default="VLM-Radiology-Agent-Framework/m3/data_prepare/experts/llama_output/llama_gen_expert_what.txt",
        help="Path to the LLaMA-generated description questions (used as-is for the description group)",
    )
    p.add_argument(
        "--out-dir",
        default="vila_m3_conversations",
        help="Directory to write output JSONs",
    )
    p.add_argument(
        "--n-paraphrases",
        type=int,
        default=10,
        help="Number of paraphrases Gemma generates per seed (grade + segmentation only)",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Model dtype. bfloat16 recommended for A100.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # ── Step 1: Load Gemma (only needed for grade + segmentation) ─────────
    model, tokenizer = load_model(args.model, dtype=dtype)

    # ── Step 2: Find fixed image ───────────────────────────────────────────
    print("\nSelecting fixed test image...")
    fixed_images = find_fixed_image(args.test_json, args.slices_dir, args.remote_prefix)

    # ── Step 3: Load LLaMA description questions ──────────────────────────
    print(f"\nLoading LLaMA description questions from: {args.llama_questions}")
    with open(args.llama_questions) as f:
        llama_questions = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(llama_questions)} description questions")

    # ── Step 4: Build all paraphrase groups ───────────────────────────────
    paraphrase_groups = []
    eval_samples = []

    # --- Description: use LLaMA questions directly (no Gemma needed) ---
    print(f"\n[description] Using {len(llama_questions)} existing LLaMA questions (no Gemma)")
    desc_queries = llama_questions
    paraphrase_groups.append(
        {
            "category": "description",
            "expected_expert": "none",
            "source": "llama_generated",
            "seed": None,
            "paraphrases": desc_queries,
            "all_queries": desc_queries,
            "n_queries": len(desc_queries),
            "n_pairs": len(desc_queries) * (len(desc_queries) - 1) // 2,
        }
    )
    for q_idx, query in enumerate(desc_queries):
        sample = build_eval_conversation(query, fixed_images)
        sample["sample_id"] = f"description_{q_idx:02d}"
        sample["category"] = "description"
        sample["expected_expert"] = "none"
        sample["group_id"] = "description"
        sample["is_seed"] = False
        sample["question"] = query
        eval_samples.append(sample)

    # --- Grade + Segmentation: use Gemma ---
    for category, cfg in GEMMA_CATEGORIES.items():
        seed = cfg["seed"]
        expected_expert = cfg["expected_expert"]
        context = cfg["context"]

        print(f"\n[{category}] Generating {args.n_paraphrases} paraphrases for: '{seed}'")

        paraphrases = generate_paraphrases(
            model, tokenizer, seed, context, n=args.n_paraphrases
        )

        print(f"  Parsed {len(paraphrases)} paraphrases:")
        for i, p in enumerate(paraphrases, 1):
            print(f"    {i:2d}. {p}")

        all_queries = [seed] + paraphrases

        paraphrase_groups.append(
            {
                "category": category,
                "expected_expert": expected_expert,
                "source": "gemma_generated",
                "seed": seed,
                "paraphrases": paraphrases,
                "all_queries": all_queries,
                "n_queries": len(all_queries),
                "n_pairs": len(all_queries) * (len(all_queries) - 1) // 2,
            }
        )

        for q_idx, query in enumerate(all_queries):
            sample = build_eval_conversation(query, fixed_images)
            sample["sample_id"] = f"{category}_{q_idx:02d}"
            sample["category"] = category
            sample["expected_expert"] = expected_expert
            sample["group_id"] = category
            sample["is_seed"] = q_idx == 0
            sample["question"] = query
            eval_samples.append(sample)

    # ── Step 5: Save outputs ───────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    groups_path = os.path.join(args.out_dir, "rc_paraphrase_groups.json")
    eval_path = os.path.join(args.out_dir, "rc_eval_samples.json")

    with open(groups_path, "w") as f:
        json.dump(paraphrase_groups, f, indent=2)

    with open(eval_path, "w") as f:
        json.dump(eval_samples, f, indent=2)

    # ── Step 6: Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Paraphrase groups : {groups_path}")
    print(f"  Eval samples      : {eval_path}")
    print()
    for g in paraphrase_groups:
        print(
            f"  [{g['category']:12s}]  {g['n_queries']} queries  "
            f"({g['n_pairs']} RC pairs)  expected_expert={g['expected_expert']}"
        )
    print(f"\n  Total eval samples: {len(eval_samples)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
