# Responsible AI Assignment 2

This repository contains a multi-part Trust and Safety assignment for toxicity detection, fairness auditing, adversarial robustness, and production guardrail design.

## Environment

- Python version: 3.12 (Google Colab runtime used during development)
- GPU used: NVIDIA Tesla T4
- OS used for authoring local files: Windows

## Project Artifacts

- Notebooks: `part_1.ipynb`, `part_2.ipynb`, `part_3.ipynb`, `part4.ipynb`, `part_5.ipynb`
- Pipeline code: `pipeline.py`
- Key checkpoints/artifacts:
  - `distilbert_toxicity_checkpoint_part_1/`
  - `best_mitigated_model_part_4/`

## How To Reproduce (Google Colab T4)

1. Open Google Colab.
2. Set runtime to GPU:
   - Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU
3. Clone this repository and move into it.
4. Install dependencies:
   - `pip install -r requirements.txt`
5. Ensure required data/artifact files are present in the working directory:
   - `jigsaw-unintended-bias-train.csv`
   - `validation.csv`
   - `train_subset_100k.csv` and `eval_subset_20k.csv` (or regenerate via Part 1)
   - Model zip/checkpoint files used by later parts
6. Run notebooks in order:
   - `part_1.ipynb`
   - `part_2.ipynb`
   - `part_3.ipynb`
   - `part4.ipynb`
   - `part_5.ipynb`

## Reproducibility Notes

- Random seed used in notebook workflows: 42
- If a notebook uses extracted model directories, unzip corresponding archives first.
- If kernel state was changed, restart runtime and rerun cells top-to-bottom.

## Quick Validation

After running all parts, verify that:

- Part 1 produces baseline model metrics and threshold analysis.
- Part 2 produces subgroup fairness metrics and disparity comparisons.
- Part 3 reports evasion attack success and poisoning impact.
- Part 4 compares mitigation techniques and saves best mitigated model.
- Part 5 reports layer-wise moderation outcomes and threshold-band tradeoff analysis.
