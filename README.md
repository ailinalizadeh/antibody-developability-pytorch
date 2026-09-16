# Antibody Developability Prediction with PyTorch

> **Portfolio status:** The implementation is complete; model performance must be generated locally before reporting numerical results.

### Windows installation note

Install the core dependencies first:

```powershell
pip install -r requirements.txt
```

Then install the TDC dataset loader without its optional dependency tree:

```powershell
pip install PyTDC==0.4.17 --no-deps
```

This repository intentionally does not fabricate performance numbers. Run `train.py` and `evaluate.py` before adding test metrics to the README or CV.


A reproducible student research project for predicting antibody developability
from paired heavy- and light-chain amino-acid sequences using a compact PyTorch
1D convolutional neural network (CNN).

## Research question

Can a compact sequence-based PyTorch model learn useful signal for antibody
developability from paired heavy- and light-chain amino-acid sequences?

This repository is intended as an educational/research portfolio project. It
does **not** claim to be a clinically validated or therapeutic-grade predictor.

## Dataset

This project uses the **SAbDab_Chen** dataset from Therapeutics Data Commons
(TDC).

- Task: binary classification
- Input: paired heavy-chain and light-chain amino-acid sequences
- Size: 2,409 antibodies
- TDC split: random train/validation/test split
- Dataset license: CC BY 3.0
- Source: https://tdcommons.ai/single_pred_tasks/develop/

The raw dataset is not committed to this repository. Run the data download
script to retrieve it through TDC.

## Model

Each antibody chain is processed independently:

1. amino-acid tokenization
2. learnable embedding
3. 1D convolution
4. ReLU activation
5. masked global max pooling
6. concatenation of heavy- and light-chain representations
7. multilayer perceptron classifier

The model returns one logit per antibody and is trained with
`BCEWithLogitsLoss`.

## Repository structure

```text
antibody-developability-pytorch/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── train.py
├── evaluate.py
├── scripts/
│   └── fetch_data.py
├── src/
│   └── antibody_developability/
│       ├── __init__.py
│       ├── data.py
│       ├── metrics.py
│       ├── model.py
│       └── utils.py
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── data/
│   ├── raw/
│   └── processed/
├── checkpoints/
└── results/
```

## Environment

Python 3.11 is recommended.

### Create a virtual environment

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Download and prepare the data

```bash
python scripts/fetch_data.py
```

Expected output files:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/data_summary.json
```

The script validates that each example contains a heavy chain, a light chain,
and a binary label.

## 2. Run tests

```bash
pytest -q
```

## 3. Train the model

```bash
python train.py --epochs 30 --batch-size 32 --learning-rate 0.001
```

Training uses the validation set for model selection. The best checkpoint is
saved to:

```text
checkpoints/best_model.pt
```

## 4. Evaluate once on the held-out test set

```bash
python evaluate.py
```

This creates:

```text
results/test_metrics.json
results/roc_curve.png
results/pr_curve.png
results/confusion_matrix.png
results/test_predictions.csv
```

## Results

**Do not fill in this section until you have actually run the project.**

| Metric | Test result |
|---|---:|
| AUROC | TBD |
| AUPRC | TBD |
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 | TBD |

After running `evaluate.py`, copy the real values from
`results/test_metrics.json` into this table.

## Scientific interpretation

The project is a sequence-based baseline. Performance on a random split does
not establish performance on prospectively generated antibodies or on a
structurally independent external dataset.

Potential next steps:

- repeated random seeds
- sequence-similarity-aware splitting
- class-threshold tuning using validation data only
- comparison with a simpler k-mer/logistic-regression baseline
- protein language model embeddings as a later extension

## Reproducibility

The project records:

- random seed
- model hyperparameters
- maximum sequence length
- validation history
- saved model checkpoint
- test predictions
- environment dependencies

PyTorch results can still vary across versions, devices, and platforms, so
software and hardware details should be recorded when reporting final results.

## Citation and data attribution

Therapeutics Data Commons (TDC), SAbDab_Chen developability dataset.

Chen X. et al. *Predicting antibody developability from sequence using machine
learning* (2020).

Dunbar J. et al. *SAbDab: the structural antibody database*. Nucleic Acids
Research (2014).

## License

Code in this repository is released under the MIT License. The dataset is not
redistributed by this repository and remains subject to its original license
and citation requirements.
