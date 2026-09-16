from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from antibody_developability.data import AntibodyDataset
from antibody_developability.metrics import classification_metrics
from antibody_developability.model import AntibodyDevelopabilityCNN
from antibody_developability.utils import choose_device, save_json


def main():
    checkpoint_path = Path("checkpoints/best_model.pt")
    test_path = Path("data/processed/test.csv")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "No checkpoint found. Train the model first with: python train.py"
        )
    if not test_path.exists():
        raise FileNotFoundError(
            "No test data found. Run: python scripts/fetch_data.py"
        )

    device = choose_device()
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]

    dataset = AntibodyDataset(test_path, max_len=int(config["max_len"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )

    model = AntibodyDevelopabilityCNN(
        embedding_dim=int(config["embedding_dim"]),
        channels=int(config["channels"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = []
    probabilities = []

    with torch.no_grad():
        for batch in loader:
            heavy = batch["heavy"].to(device)
            light = batch["light"].to(device)
            logits = model(heavy, light)
            probs = torch.sigmoid(logits)

            labels.extend(batch["label"].numpy().astype(int).tolist())
            probabilities.extend(probs.cpu().numpy().tolist())

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)

    metrics = classification_metrics(labels, probabilities)
    save_json(metrics, results_dir / "test_metrics.json")

    predictions_df = dataset.df.copy()
    predictions_df["predicted_probability"] = probabilities
    predictions_df["predicted_label"] = predictions
    predictions_df.to_csv(results_dir / "test_predictions.csv", index=False)

    RocCurveDisplay.from_predictions(labels, probabilities)
    plt.title("ROC curve — held-out test set")
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curve.png", dpi=200)
    plt.close()

    PrecisionRecallDisplay.from_predictions(labels, probabilities)
    plt.title("Precision–recall curve — held-out test set")
    plt.tight_layout()
    plt.savefig(results_dir / "pr_curve.png", dpi=200)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(labels, predictions)
    plt.title("Confusion matrix — held-out test set")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print("\nSaved evaluation outputs to results/")


if __name__ == "__main__":
    main()
