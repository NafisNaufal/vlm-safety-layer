#!/usr/bin/env python3
"""
VILA-M3 training data preparation — BraTS, VISTA3D, CXR.

Conversation formats match the original VILA-M3 data_prepare scripts exactly:
  BraTS   : 3 types per slice — what (2-turn), grade (4-turn), seg (4-turn)
  VISTA3D : 2 types per slice — seg (2-turn, routing in answer), what (4-turn)
  CXR     : 1 type per image — always 4-turn with <CXR()> trigger

Phases (--phase):
  paraphrases   Gemma generates paraphrase sets for all seed questions → questions.json
  extract       VISTA3D NIfTI volumes → 2D PNG slices + vista3d_meta.json
  txrv          TorchXRayVision inference on MIMIC-CXR images → txrv_preds/*.json
  conversations Build VILA training JSONs from all prepared data
  all           Run all four phases in order

Example:
  python prepare_training_data.py --phase all \\
    --brats_slices_dir /path/to/BraTS/vila_m3_slices \\
    --vista3d_dir /path/to/vista3d \\
    --cxr_dir /path/to/mimic-cxr \\
    --out_dir /path/to/training_data
"""

import argparse
import json
import os
import random
import re
import uuid
from pathlib import Path

import torch
from tqdm import tqdm

random.seed(42)

# ─── VILA-M3 model cards (verbatim from expert_utils.py) ─────────────────────

MODEL_CARDS = (
    "Here is a list of available expert models:\n"
    "<BRATS(args)> Modality: MRI, Task: segmentation, Overview: A pre-trained model for volumetric (3D) "
    "segmentation of brain tumor subregions from multimodal MRIs based on BraTS 2018 data, Accuracy: "
    "Tumor core (TC): 0.8559 - Whole tumor (WT): 0.9026 - Enhancing tumor (ET): 0.7905 - Average: 0.8518, "
    "Valid args are: None\n"
    "<VISTA3D(args)> Modality: CT, Task: segmentation, Overview: domain-specialized interactive foundation "
    "model developed for segmenting and annotating human anatomies with precision, Accuracy: 127 organs: "
    "0.792 Dice on average, Valid args are: 'everything', 'hepatic tumor', 'pancreatic tumor', 'lung tumor', "
    "'bone lesion', 'organs', 'cardiovascular', 'gastrointestinal', 'skeleton', or 'muscles'\n"
    "<VISTA2D(args)> Modality: cell imaging, Task: segmentation, Overview: model for cell segmentation, "
    "which was trained on a variety of cell imaging outputs, including brightfield, phase-contrast, "
    "fluorescence, confocal, or electron microscopy, Accuracy: Good accuracy across several cell imaging "
    "datasets, Valid args are: None\n"
    "<CXR(args)> Modality: chest x-ray (CXR), Task: classification, Overview: pre-trained model which are "
    "trained on large cohorts of data, Accuracy: Good accuracy across several diverse chest x-rays datasets, "
    "Valid args are: None\n"
    "Give the model <NAME(args)> when selecting a suitable expert model.\n"
)

# ─── Seed questions for each conversation type ────────────────────────────────

SEEDS = {
    # BraTS what: plain descriptive, no routing (matches llama_gen_expert_what.txt style)
    "brats_what": {
        "context": "brain MRI general description — model identifies the modality, no expert routing needed",
        "questions": [
            "What can be seen in this scan?",
            "Can you describe what's in the image?",
            "What does the image depict?",
        ],
    },
    # BraTS grade: triggers <BRATS()>, followed by grade answer
    "brats_grade": {
        "context": "brain MRI glioma grading — model should output <BRATS()> routing token",
        "questions": [
            "What glioma grade is shown in the image?",
            "Does the imaging evidence suggest a high-grade or low-grade glioma?",
        ],
    },
    # BraTS seg: triggers <BRATS()> with seg-specific phrasing, followed by seg answer
    "brats_seg": {
        "context": "brain MRI tumor segmentation — model should output <BRATS()> routing token",
        "questions": [
            "Segment any tumors in the image.",
            "Please outline the boundaries of any intracranial tumors.",
        ],
    },
    # VISTA3D seg: 2-turn — answer embeds <VISTA3D(...)> routing token directly
    "vista3d_seg": {
        "context": "CT organ/tumor segmentation — routing token <VISTA3D(...)> goes inside the GPT answer",
        "questions": [
            "Segment the organs in this CT scan.",
            "Outline all anatomical structures visible in this abdominal CT.",
            "Can you identify and delineate the organs in this CT image?",
        ],
    },
    # VISTA3D what: 4-turn — gpt turn 1 triggers VISTA3D, gpt turn 2 describes result
    "vista3d_what": {
        "context": "CT organ description — model triggers VISTA3D expert then describes what was found",
        "questions": [
            "What can be seen in this CT scan?",
            "What anatomical structures are present in this CT image?",
            "Can you describe the contents of this abdominal CT?",
        ],
    },
    # CXR: 4-turn — always triggers <CXR()>, predictions fed back, model gives clinical summary
    "cxr": {
        "context": "chest X-ray pathology classification — model triggers <CXR()> then summarizes findings",
        "questions": [
            "Describe any findings in this chest X-ray.",
            "Are there any abnormalities visible in this chest X-ray?",
            "What pathologies, if any, are present in this CXR?",
        ],
    },
}

VISTA3D_TASK_MAP = {
    "Task03_Liver": {
        "organ": "liver",
        "vista_arg": "hepatic tumor",
        "label_vals":   {1: "liver",    2: "hepatic tumor"},
        "label_colors": {1: (220, 80, 60), 2: (255, 220, 50)},
        "color_names":  {1: "red",      2: "yellow"},
    },
    "Task07_Pancreas": {
        "organ": "pancreas",
        "vista_arg": "pancreatic tumor",
        "label_vals":   {1: "pancreas", 2: "pancreatic tumor"},
        "label_colors": {1: (80, 160, 220), 2: (255, 220, 50)},
        "color_names":  {1: "blue",     2: "yellow"},
    },
    "Task09_Spleen": {
        "organ": "spleen",
        "vista_arg": "organs",
        "label_vals":   {1: "spleen"},
        "label_colors": {1: (100, 200, 100)},
        "color_names":  {1: "green"},
    },
}

# ─── Gemma paraphrase generation ─────────────────────────────────────────────

PARAPHRASE_PROMPT = """\
You are a medical imaging assistant helping generate training data for a VLM routing system.

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


def _parse_numbered_list(text: str, expected: int) -> list:
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[\.\)\:]?\s+(.+)$", line)
        if m:
            results.append(m.group(1).strip())
    return results[:expected]


def load_gemma(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading Gemma: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print("Gemma loaded.")
    return model, tokenizer


def generate_paraphrases(model, tokenizer, seed: str, context: str, n: int) -> list:
    prompt = PARAPHRASE_PROMPT.format(n=n, context=context, seed=seed)
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    paraphrases = _parse_numbered_list(raw, expected=n)
    if len(paraphrases) < n:
        print(f"  Warning: got {len(paraphrases)}/{n} paraphrases for: {seed!r}")
    return paraphrases


def phase_paraphrases(args):
    out_path = Path(args.out_dir) / "questions.json"
    if out_path.exists() and not args.overwrite:
        print(f"Skipping — {out_path} already exists (use --overwrite to regenerate)")
        return

    model, tokenizer = load_gemma(args.gemma_model)
    questions = {}

    for key, info in SEEDS.items():
        context = info["context"]
        all_q = list(info["questions"])
        for seed in tqdm(info["questions"], desc=f"Paraphrasing {key}"):
            paraphrases = generate_paraphrases(model, tokenizer, seed, context, args.n_paraphrases)
            all_q.extend(paraphrases)
        seen, unique = set(), []
        for q in all_q:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique.append(q)
        questions[key] = unique
        print(f"  {key}: {len(unique)} questions")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"\nSaved questions → {out_path}")


# ─── VISTA3D slice extraction ─────────────────────────────────────────────────

def phase_extract_vista3d(args):
    import numpy as np
    from PIL import Image

    meta_path = Path(args.out_dir) / "vista3d_meta.json"
    if meta_path.exists() and not args.overwrite:
        print(f"Skipping — {meta_path} already exists (use --overwrite to redo)")
        return

    try:
        from monai.transforms import LoadImage, Orientation
    except ImportError:
        raise RuntimeError("MONAI required: pip install monai")

    slices_dir = Path(args.out_dir) / "vista3d_slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    loader = LoadImage(image_only=True, ensure_channel_first=True)
    orient = Orientation(axcodes="RAS")

    # Abdomen soft-tissue CT window: center=50, width=400
    CT_MIN, CT_MAX = -150, 250

    all_meta = []

    for task_name, task_info in VISTA3D_TASK_MAP.items():
        task_dir = Path(args.vista3d_dir) / task_name
        if not task_dir.exists():
            print(f"WARNING: {task_dir} not found, skipping")
            continue

        img_files = sorted(p for p in (task_dir / "imagesTr").glob("*.nii.gz") if not p.name.startswith("._"))
        print(f"\n{task_name}: {len(img_files)} volumes")
        task_slices_dir = slices_dir / task_name
        task_slices_dir.mkdir(exist_ok=True)

        for img_path in tqdm(img_files, desc=f"Extracting {task_name}"):
            lbl_path = task_dir / "labelsTr" / img_path.name
            if not lbl_path.exists():
                continue
            try:
                img_vol = orient(loader(str(img_path))).numpy()[0]   # H, W, D
                lbl_vol = orient(loader(str(lbl_path))).numpy()[0]
            except Exception as e:
                print(f"  Error {img_path.name}: {e}")
                continue

            stem = img_path.name.replace(".nii.gz", "")

            for z in range(img_vol.shape[2]):
                lbl_slice = lbl_vol[:, :, z].astype(np.int32)  # MONAI loads as float32
                if lbl_slice.max() == 0:
                    continue

                img_u8 = np.clip(
                    (img_vol[:, :, z] - CT_MIN) / (CT_MAX - CT_MIN) * 255, 0, 255
                ).astype(np.uint8)

                h, w = lbl_slice.shape
                lbl_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                for val, color in task_info["label_colors"].items():
                    lbl_rgb[lbl_slice == val] = color

                sid = f"{stem}_z{z:03d}"
                img_out = str(task_slices_dir / f"{sid}_img.png")
                lbl_out = str(task_slices_dir / f"{sid}_label.png")
                Image.fromarray(img_u8).save(img_out)
                Image.fromarray(lbl_rgb).save(lbl_out)

                present = [v for v in np.unique(lbl_slice) if v > 0 and v in task_info["label_vals"]]
                label_names = [task_info["label_vals"][v] for v in present]
                # "red: liver, yellow: hepatic tumor" — matches original m['label_colors'] format
                label_colors_str = ", ".join(
                    f"{task_info['color_names'][v]}: {task_info['label_vals'][v]}"
                    for v in present
                )
                all_meta.append({
                    "image": img_out,
                    "label": lbl_out,
                    "task": task_name,
                    "organ": task_info["organ"],
                    "vista_arg": task_info["vista_arg"],
                    "label_names": label_names,
                    "label_colors_str": label_colors_str,
                })

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\nExtracted {len(all_meta)} VISTA3D slices → {meta_path}")


# ─── TorchXRayVision inference ────────────────────────────────────────────────

def phase_txrv(args):
    import numpy as np
    from PIL import Image

    try:
        import torchxrayvision as xrv
    except ImportError:
        raise RuntimeError("TorchXRayVision required: pip install torchxrayvision")

    preds_dir = Path(args.out_dir) / "txrv_preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    jpg_files = list(Path(args.cxr_dir).joinpath("files").rglob("*.jpg"))
    print(f"Found {len(jpg_files)} CXR JPEGs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
    model.eval()

    for jpg_path in tqdm(jpg_files, desc="TorchXRayVision"):
        out_path = preds_dir / f"{jpg_path.stem}.json"
        if out_path.exists() and not args.overwrite:
            continue
        try:
            img_np = np.array(Image.open(jpg_path).convert("L")).astype(np.float32)
            img_np = xrv.datasets.normalize(img_np, 255)[None]
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)
            img_tensor = torch.nn.functional.interpolate(
                img_tensor, size=(224, 224), mode="bilinear", align_corners=False
            )
            with torch.no_grad():
                preds = model(img_tensor).cpu().numpy()[0]
            pred_dict = {label: float(p) for label, p in zip(model.pathologies, preds) if label}
            with open(out_path, "w") as f:
                json.dump(pred_dict, f)
        except Exception as e:
            print(f"  Error {jpg_path.name}: {e}")

    print(f"Predictions saved → {preds_dir}")


# ─── Conversation builders ────────────────────────────────────────────────────

def _save_three_way(train_c: list, val_c: list, test_c: list, out_dir: str, prefix: str):
    for split, data in [("train", train_c), ("val", val_c), ("test", test_c)]:
        path = Path(out_dir) / f"{prefix}_{split}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"  {prefix} {split}: {len(data):,} → {path}")


def _load_questions(out_dir: str) -> dict:
    path = Path(out_dir) / "questions.json"
    if not path.exists():
        raise FileNotFoundError(f"Run --phase paraphrases first (missing {path})")
    with open(path) as f:
        return json.load(f)


def build_brats(args, questions: dict) -> tuple:
    """
    3 conversation types per slice (matching notebook cell 14-15 exactly):
      1. what  — 2-turn, no model_list, no <image> tokens, 1 random what_q per slice
      2. grade — 4-turn, ALL grade_qs per slice
      3. seg   — 4-turn, ALL seg_qs per slice, seg-specific trigger

    Reads extracted_slices_meta.json from brats_slices_dir for proper train/val/test split,
    HGG/LGG label, and image paths.

    Returns (train_convs, val_convs, test_convs).
    """
    slices_dir = Path(args.brats_slices_dir)
    meta_path = slices_dir / "extracted_slices_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"BraTS metadata not found: {meta_path}\n"
            "Run the BraTS slice extraction notebook first."
        )
    with open(meta_path) as f:
        all_slice_meta = json.load(f)

    what_qs  = questions["brats_what"]
    grade_qs = questions["brats_grade"]
    seg_qs   = questions["brats_seg"]

    seg_result = (
        "The results are <image>. The colors in this image describe\n"
        "yellow and red: tumor core, only yellow: enhancing tumor, all colors: whole tumor\n"
    )
    mri_header = (
        MODEL_CARDS
        + "T1(contrast enhanced): <image>, T1: <image>, T2: <image>, FLAIR: <image> "
        "These are different MRI modalities.\n"
    )

    train_convs, val_convs, test_convs = [], [], []

    for meta in tqdm(all_slice_meta, desc="BraTS conversations"):
        is_hgg = "HGG" in meta["orig_images"][0]
        grade_ans = (
            "The scan indicates high grade glioma."
            if is_hgg else "The scan indicates lower grade glioma."
        )
        split = meta.get("split", "train")

        all_images = [
            str(slices_dir / meta["image"][0]),
            str(slices_dir / meta["image"][1]),
            str(slices_dir / meta["image"][2]),
            str(slices_dir / meta["image"][3]),
            str(slices_dir / meta["label"]),
        ]

        target = train_convs if split == "train" else (val_convs if split == "val" else test_convs)

        # 1. What — 2-turn, no model_list, no <image> tokens
        target.append({
            "images": all_images,
            "conversations": [
                {"from": "human", "value": random.choice(what_qs)},
                {"from": "gpt",   "value": "This is a brain MRI scan."},
            ],
        })

        # 2. Grade — 4-turn, one entry per question
        for q in grade_qs:
            target.append({
                "images": all_images,
                "conversations": [
                    {"from": "human", "value": mri_header + q},
                    {"from": "gpt",   "value": "This looks like an MRI image sequence. Let me first trigger <BRATS()>."},
                    {"from": "human", "value": seg_result + f"Use this result to respond to this prompt:\n{q}."},
                    {"from": "gpt",   "value": grade_ans},
                ],
            })

        # 3. Seg — 4-turn, one entry per question
        for q in seg_qs:
            target.append({
                "images": all_images,
                "conversations": [
                    {"from": "human", "value": mri_header + q},
                    {"from": "gpt",   "value": "I segmented any brain tumors using <BRATS()>."},
                    {"from": "human", "value": seg_result + f"Use this result to respond to this prompt:\n{q}."},
                    {"from": "gpt",   "value": "All tumors were segmented"},
                ],
            })

    return train_convs, val_convs, test_convs


def build_vista3d(args, questions: dict) -> tuple:
    """
    ALL questions of both types per slice (same approach as BraTS grade/seg):
      seg  — 2-turn per question, routing token <VISTA3D(...)> embedded in GPT answer
      what — 4-turn per question, GPT triggers VISTA3D then describes result

    Volume-level 3-way split: slices from the same CT volume always stay in the same
    split (train/val/test), preventing data leakage across splits.

    Returns (train_convs, val_convs, test_convs).
    """
    meta_path = Path(args.out_dir) / "vista3d_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run --phase extract first (missing {meta_path})")
    with open(meta_path) as f:
        meta = json.load(f)

    seg_qs  = questions["vista3d_seg"]
    what_qs = questions["vista3d_what"]

    # Group conversations by volume (strip _z{digits}_img.png suffix to get volume id)
    volume_convs: dict = {}

    for m in tqdm(meta, desc="VISTA3D conversations"):
        img_name = Path(m["image"]).name
        vol_stem = re.sub(r'_z\d+_img\.png$', '', img_name)
        vol_key = f"{m['task']}/{vol_stem}"

        if vol_key not in volume_convs:
            volume_convs[vol_key] = []

        vista_arg = m["vista_arg"]
        label_colors_str = m["label_colors_str"]

        # Seg — 2-turn, ALL seg questions per slice (routing token in answer)
        for q in seg_qs:
            volume_convs[vol_key].append({
                "id": str(uuid.uuid4()),
                "images": [m["image"]],
                "conversations": [
                    {"from": "human", "value": MODEL_CARDS + f" <image>This is a CT image.\n{q}"},
                    {"from": "gpt",   "value": f"I segmented the {m['organ']} using <VISTA3D({vista_arg})>."},
                ],
            })

        # What — 4-turn, ALL what questions per slice (trigger in GPT turn 1)
        for q in what_qs:
            volume_convs[vol_key].append({
                "id": str(uuid.uuid4()),
                "images": [m["image"], m["label"]],
                "conversations": [
                    {"from": "human", "value": MODEL_CARDS + f" <image>This is a CT image.\n{q}"},
                    {"from": "gpt",   "value": f"This looks like a CT image. Let me trigger <VISTA3D({vista_arg})>. "},
                    {"from": "human", "value": (
                        f"The results are <image>. "
                        f"The colors in this image describe {label_colors_str}. "
                        f"Use this result to respond to this prompt:\n{q}."
                    )},
                    {"from": "gpt",   "value": (
                        f"This a CT image. "
                        f"It contains several anatomical structures such as identified by VISTA3D: "
                        f"{label_colors_str}."
                    )},
                ],
            })

    # Volume-level split: shuffle volumes, then assign all their slices to the same split
    volumes = list(volume_convs.keys())
    random.shuffle(volumes)
    n_test = max(1, int(len(volumes) * args.test_frac))
    n_val  = max(1, int(len(volumes) * args.test_frac))

    test_vols  = volumes[:n_test]
    val_vols   = volumes[n_test:n_test + n_val]
    train_vols = volumes[n_test + n_val:]

    train_c = [c for v in train_vols for c in volume_convs[v]]
    val_c   = [c for v in val_vols   for c in volume_convs[v]]
    test_c  = [c for v in test_vols  for c in volume_convs[v]]

    print(f"  VISTA3D volumes: {len(train_vols)} train / {len(val_vols)} val / {len(test_vols)} test")
    return train_c, val_c, test_c


def build_cxr(args, questions: dict) -> tuple:
    """
    ALL cxr questions per image (same approach as BraTS grade/seg):
      4-turn per question — <CXR()> trigger in GPT turn 1, predictions fed back, model summarizes findings

    Using all paraphrases (not random.choice) enables RC evaluation on the test split.
    Image-level 3-way split (CXR are independent 2D images, no volumetric leakage concern).

    Returns (train_convs, val_convs, test_convs).
    """
    preds_dir = Path(args.out_dir) / "txrv_preds"
    cxr_files_dir = Path(args.cxr_dir) / "files"
    if not preds_dir.exists():
        raise FileNotFoundError(f"Run --phase txrv first (missing {preds_dir})")

    cxr_qs = questions["cxr"]
    all_convs = []

    # Build dicom_id → jpg_path index once (avoid O(n²) rglob per file)
    print("Building CXR file index...")
    jpg_index = {p.stem: p for p in cxr_files_dir.rglob("*.jpg")}
    print(f"  Indexed {len(jpg_index)} JPEGs")

    for pred_file in tqdm(sorted(preds_dir.glob("*.json")), desc="CXR conversations"):
        jpg_path = jpg_index.get(pred_file.stem)
        if jpg_path is None:
            continue

        with open(pred_file) as f:
            preds = json.load(f)

        pred_str = "".join(
            f"{k.lower().replace('_', ' ')}: {'yes' if v > 0.5 else 'no'}\n"
            for k, v in preds.items() if k
        )
        positives = [k.lower().replace("_", " ") for k, v in preds.items() if k and v > 0.5]
        answer = (
            f"The chest X-ray shows findings consistent with: {', '.join(positives)}."
            if positives else "No significant abnormalities detected in this chest X-ray."
        )

        # ALL cxr questions per image — enables RC eval on the test split
        for q in cxr_qs:
            all_convs.append({
                "id": str(uuid.uuid4()),
                "images": [str(jpg_path)],
                "conversations": [
                    {"from": "human", "value": MODEL_CARDS + f"<image> This is a CXR image.\n{q}"},
                    {"from": "gpt",   "value": "This looks like an chest x-ray. Let me first trigger <CXR()>."},
                    {"from": "human", "value": (
                        f"The resulting predictions are:\n{pred_str}. "
                        f"Analyze the image and take these predictions into account when responding to this prompt:\n{q}"
                    )},
                    {"from": "gpt",   "value": answer},
                ],
            })

    # Image-level 3-way split (shuffle conversations)
    random.shuffle(all_convs)
    n_test = max(1, int(len(all_convs) * args.test_frac))
    n_val  = max(1, int(len(all_convs) * args.test_frac))
    test_c  = all_convs[:n_test]
    val_c   = all_convs[n_test:n_test + n_val]
    train_c = all_convs[n_test + n_val:]
    return train_c, val_c, test_c


def phase_conversations(args):
    questions = _load_questions(args.out_dir)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",")]

    if "brats" in datasets:
        train_c, val_c, test_c = build_brats(args, questions)
        n = args.n_samples_brats
        if n:
            # sample proportionally across splits
            total = len(train_c) + len(val_c) + len(test_c)
            if total > n:
                train_c = random.sample(train_c, int(n * len(train_c) / total))
                val_c   = random.sample(val_c,   int(n * len(val_c)   / total))
                test_c  = random.sample(test_c,  int(n * len(test_c)  / total))
        for split, data in [("train", train_c), ("val", val_c), ("test", test_c)]:
            path = Path(args.out_dir) / f"brats_{split}.json"
            with open(path, "w") as f:
                json.dump(data, f)
            print(f"  BraTS {split}: {len(data):,} → {path}")

    if "vista3d" in datasets:
        train_c, val_c, test_c = build_vista3d(args, questions)
        n = args.n_samples_vista3d
        if n:
            total = len(train_c) + len(val_c) + len(test_c)
            if total > n:
                train_c = random.sample(train_c, int(n * len(train_c) / total))
                val_c   = random.sample(val_c,   int(n * len(val_c)   / total))
                test_c  = random.sample(test_c,  int(n * len(test_c)  / total))
        print(f"VISTA3D: {len(train_c)+len(val_c)+len(test_c):,} conversations total")
        _save_three_way(train_c, val_c, test_c, args.out_dir, "vista3d")

    if "cxr" in datasets:
        train_c, val_c, test_c = build_cxr(args, questions)
        n = args.n_samples_cxr
        if n:
            total = len(train_c) + len(val_c) + len(test_c)
            if total > n:
                train_c = random.sample(train_c, int(n * len(train_c) / total))
                val_c   = random.sample(val_c,   int(n * len(val_c)   / total))
                test_c  = random.sample(test_c,  int(n * len(test_c)  / total))
        print(f"CXR: {len(train_c)+len(val_c)+len(test_c):,} conversations total")
        _save_three_way(train_c, val_c, test_c, args.out_dir, "cxr")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True,
                        choices=["paraphrases", "extract", "txrv", "conversations", "all"])
    parser.add_argument("--out_dir", default="./training_data")
    parser.add_argument("--datasets", default="brats,vista3d,cxr",
                        help="Comma-separated subset to process in conversations phase")

    # Data paths
    parser.add_argument("--brats_slices_dir", help="BraTS PNG slices dir (vila_m3_slices/) — must contain extracted_slices_meta.json")
    parser.add_argument("--vista3d_dir",      help="VISTA3D root (contains Task03_Liver/ etc.)")
    parser.add_argument("--cxr_dir",          help="MIMIC-CXR mini dir (contains files/ and metadata.csv)")

    # Gemma
    parser.add_argument("--gemma_model",   default="google/gemma-4-E4B-it")
    parser.add_argument("--n_paraphrases", type=int, default=10)

    # Sampling (0 = use all)
    parser.add_argument("--n_samples_brats",   type=int, default=0)
    parser.add_argument("--n_samples_vista3d", type=int, default=0)
    parser.add_argument("--n_samples_cxr",     type=int, default=0)
    parser.add_argument("--test_frac", type=float, default=0.1)

    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.n_samples_brats   == 0: args.n_samples_brats   = None
    if args.n_samples_vista3d == 0: args.n_samples_vista3d = None
    if args.n_samples_cxr     == 0: args.n_samples_cxr     = None

    phases = ["paraphrases", "extract", "txrv", "conversations"] if args.phase == "all" else [args.phase]

    for phase in phases:
        print(f"\n{'='*60}\nPhase: {phase}\n{'='*60}")
        if phase == "paraphrases":
            phase_paraphrases(args)
        elif phase == "extract":
            phase_extract_vista3d(args)
        elif phase == "txrv":
            phase_txrv(args)
        elif phase == "conversations":
            phase_conversations(args)

    print("\nAll done.")


if __name__ == "__main__":
    main()
