# Concept bottleneck model for contrast-enhanced mammography (CDD-CESM)

Code, report-derived concept labels and saved predictions for the paper:

> A. K. Prajapati, A. Khosla. *Interpretable diagnosis of contrast-enhanced mammography with a concept
> bottleneck model trained on report-derived lexicon descriptors.* Submitted to *Computers in Biology and
> Medicine*, 2026.

The model predicts 35 BI-RADS / CEM-lexicon descriptors (16 low-energy, 19 recombined-image) from each image
and classifies each breast as normal, benign or malignant **only from those descriptors**. It is compared with
the network's own black-box class head under patient-grouped five-fold cross-validation on the public
CDD-CESM dataset (326 patients, 566 breasts, 2,006 images).

## What is in this repository
| Folder | Content |
|---|---|
| `labels/` | The 35 concept labels for all 2,006 images and 566 breasts, the excluded cells and the annotation guideline (CC BY 4.0). Start here if you only need the labels. |
| `pipeline/` | `build_tables.py`: builds the image, breast and fold tables from the dataset. |
| `c1_concepts/` | Concept extractor, training script, every analysis script, and the concept targets used for training. |
| `c1_concepts/validation/` | Annotation guideline, reference labels of the extractor's development and held-out test samples, validation reports. |
| `c1_concepts/runs/` | **Saved out-of-fold predictions** (`oof_fold*.csv`) and run settings (`args.txt`) of every run in the paper, plus the result files (`*.md`). Every number in the paper can be recomputed from these files on a CPU, without retraining. |
| `weights/` | Empty. Put the Mammo-CLIP weights here (see below). |

Not included: images, the full report text (the validation files quote short report phrases), trained model
weights, and the third-party Mammo-CLIP weights.

## Data
CDD-CESM (Khaled et al., *Scientific Data* 9:122, 2022; The Cancer Imaging Archive, DOI 10.7937/29kw-ae92,
CC BY 4.0):
1. Download the images from the TCIA collection page.
2. Get the dataset authors' code repository (linked from the TCIA page and the *Scientific Data* paper). It
   provides `data/train_set_All.csv`, `data/test_set_All.csv`, `data/radiology_segmentations.csv` and
   `reports/reports.csv`.
3. Put all images (`*.jpg`) into `data/images/All/` inside that repository.
4. Point the scripts to it: `set CDDCESM_ROOT=<path>` (Windows) or `export CDDCESM_ROOT=<path>` (Linux/macOS).

## Setup
Python 3.10 and the packages in `requirements.txt` (the paper used PyTorch 2.11.0 with CUDA 12.8):
```
pip install -r requirements.txt
```
Mammo-CLIP image encoder: download `b2-model-best-epoch-10.tar` from the official Mammo-CLIP repository
(https://github.com/batmanlab/Mammo-CLIP) under its licence, and place it at
`weights/mammoclip/b2-model-best-epoch-10.tar`. The checkpoint is loaded with a restricted unpickler
(`c1_concepts/safe_ckpt.py`).

## Reproduce
```
python pipeline/build_tables.py                         # tables (needs CDDCESM_ROOT)
python c1_concepts/build_concept_targets.py             # optional: rebuild the concept targets from the reports
cd c1_concepts
python -u train_backbone.py --backbone mammoclip_b2 --width 640 --height 1024 --batch 6 --run b2_mammoclip_640x1024
python -u train_backbone.py --backbone mammoclip_b2 --width 640 --height 1024 --batch 6 --seed 1 --run b2_mammoclip_640x1024_s1
python -u train_backbone.py --backbone mammoclip_b2 --width 640 --height 1024 --batch 6 --seed 2 --run b2_mammoclip_640x1024_s2
```
Training saves a checkpoint after every epoch and resumes from it when rerun. One five-fold run took about
1.5 h on an 8 GB laptop GPU. The saved predictions in `c1_concepts/runs/` let you skip training entirely.

| Paper result | Script (in `c1_concepts/`) | Output (in `c1_concepts/runs/`) |
|---|---|---|
| Table 2: concept model vs fitted and averaged black box; per seed; per machine; label-only network | `matched_black_box.py` | `MATCHED_BLACK_BOX.md` |
| Independent recomputation of the main comparison (imports no project code) | `audit_matched_baseline.py <path to this repository>` | prints the result; saved as `AUDIT_MATCHED_BASELINE.txt` |
| Calibration (Brier score, ECE) | `calibration.py` | `CALIBRATION.md` |
| Breast-label retraining (`train_backbone.py --label-source breast`, runs `b2_breastlabel_s0/1/2`) | `breast_label_check.py` | `BREAST_LABEL.md` |
| Concept AUCs, confusion matrices, head weights | `final_numbers.py` | `FACTS.md` (written to `../paper_c1/`) |
| Concept ceiling | `concept_ceiling.py` | `../concept_ceiling.md` |
| Interventions and operating points | `g6_and_clinical.py` | `G6_CLINICAL.md` |
| Grad-CAM localisation against lesion outlines | `localisation.py` | `b2_mammoclip_640x1024/LOCALISATION.md` |
| Image-level leakage and seed ensemble | `leakage_and_seeds.py` | `STEP1_SEEDS_LEAKAGE.md` |
| Residual iodine check (LE-only models) | `analyze_g4.py` | `G4_CROSS_ENERGY.md` |
| Published task definitions; binary LE model | `same_task_metrics.py`, `analyze_g2b.py` | `SAME_TASK.md`, `g2b_mammoclip_binary_LE/G2B.md` |
| Model variants (Table 4), four-view attention head | `compare_runs.py`, `fusion.py` | `COMPARISON*.md`, `b2_mammoclip_640x1024/FUSION.md` |
| Extractor validation | `concepts.py`, `make_validation_sample.py`, `findings_crosscheck.py` | `validation/` |

Run names: `b1` = ImageNet EfficientNet-B0 at 512×768; `b2_mammoclip_640x1024` (+`_s1`, `_s2`) = the main
model; `b2_mammoclip_640x1024_imagesplit` = leakage experiment; `b4_dinov2b_518x770` = DINOv2-B;
`b5_mammoclip_912x1520` = **Mammo-CLIP B2** (not B5) at its native 912×1520 resolution; `b6_mammoclip_labelonly`
= no concept loss; `g4_LE_only`, `g4_LE_cross` = LE-only models; `g2b_mammoclip_binary_LE` = binary LE model;
`b2_breastlabel_s*` = breast-label retraining; `b0` = an early pilot not reported in the paper.

## Use of AI tools
The code was written and run under the authors' direction with an AI coding assistant (Claude Code,
Anthropic); the reference labels used to validate the extractor were produced by the same assistant. Details are in
Sections 3.3 and 4.7 of the paper. No language model is part of the
model or the concept extractor.

## Licences
- Code: MIT (`LICENSE`).
- Concept labels (`labels/`, `c1_concepts/concept*.csv`, `c1_concepts/validation/gold_*.txt`): CC BY 4.0
  (`labels/LICENSE.md`). They are derived from CDD-CESM, which is CC BY 4.0: please cite the dataset too.
- Mammo-CLIP weights are not included; they are governed by their own licence.

## Citation
See `CITATION.cff`. Please also cite CDD-CESM (Khaled et al., 2022).
