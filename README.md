# vlm-safety-layer

Research code for a decoupled expert router for [VILA-M3](https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework),
a radiology vision-language model that calls specialist models (BraTS MRI segmentation, VISTA3D CT segmentation,
chest X-ray classification) by emitting routing tokens during generation.

The router is a small module that reads the image and the question together and decides which expert to call,
independently of the generative model's output:

```
image -> frozen CLIP / BiomedCLIP image encoder -> f_img (512)
text  -> frozen CLIP / BiomedCLIP text encoder  -> f_txt (512)
concat (1024) -> 2-layer MLP -> {BRATS, VISTA3D, CXR}
```

This repository contains the data preparation, router training and routing-evaluation scripts. It is research
code from a student project at Universitas Brawijaya and is shared as-is.

## Contents

| Path | Purpose |
|---|---|
| `data_prep/prepare_training_data.py` | Builds VILA-style training conversations for BraTS, VISTA3D and CXR (paraphrase generation, slice extraction, conversation JSONs). |
| `data_prep/train_router.py` | Trains the MLP router on cached frozen-encoder features. |
| `data_prep/eval_router.py` | Evaluates the router on the same eval samples used for the VILA-M3 baseline. |
| `data_prep/rc_eval_unified.py` | Routing-decision evaluation of VILA-M3 itself (routing consistency and entropy across paraphrased questions). |
| `brats2018/generate_rc_eval.py` | Generates paraphrase groups for the routing-consistency evaluation. |
| `brats2018/rc_eval_pipeline.py` | End-to-end routing-consistency pipeline on BraTS 2018. |

## Not included

- Datasets (BraTS 2018, VISTA3D, MIMIC-CXR). Obtain them from their original sources and follow their licenses.
- Model weights and checkpoints (VILA-M3, BiomedCLIP, trained routers).
- The VILA-M3 framework itself. Install it separately.

Hard-coded example paths in the scripts were replaced with `/path/to/...` placeholders. Pass your own paths
through the command-line arguments described in each script's docstring.
