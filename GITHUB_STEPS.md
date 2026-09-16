# GitHub publishing steps

## Before publishing

Run the project locally and make sure these commands work:

```bash
python scripts/fetch_data.py
pytest -q
python train.py --epochs 30
python evaluate.py
```

Then update the Results table in `README.md` with your **real** test metrics.

## Initialize Git locally

From inside the project folder:

```bash
git init
git branch -M main
git add .
git commit -m "Initialize antibody developability project"
```

## Create the GitHub repository

On GitHub:

1. Click **+**
2. Click **New repository**
3. Repository name: `antibody-developability-pytorch`
4. Description: `PyTorch sequence model for antibody developability prediction using TDC SAbDab_Chen`
5. Visibility: **Public**
6. Do **not** initialize with README, `.gitignore`, or license because those files already exist locally.
7. Click **Create repository**

Copy the HTTPS remote URL GitHub shows you.

## Connect local project to GitHub

Replace `YOUR-USERNAME` with your actual GitHub username:

```bash
git remote add origin https://github.com/YOUR-USERNAME/antibody-developability-pytorch.git
git push -u origin main
```

## Recommended later commits

Do not fake commit history. Make real commits as you work, for example:

```text
Add TDC data loading and validation
Implement amino-acid tokenizer
Add paired-chain CNN model
Add training and validation pipeline
Add held-out test evaluation
Add model tests
Document dataset provenance and limitations
Add real experiment results
```
