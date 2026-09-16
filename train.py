from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from antibody_developability.data import AntibodyDataset
from antibody_developability.metrics import classification_metrics
from antibody_developability.model import AntibodyDevelopabilityCNN
from antibody_developability.utils import choose_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    return parser.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    losses = []
    y_true = []
    probabilities = []

    for batch in loader:
        heavy = batch["heavy"].to(device)
        light = batch["light"].to(device)
        labels = batch["label"].to(device)

        if training:
            optimizer.zero_grad()

        logits = model(heavy, light)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        y_true.extend(labels.detach().cpu().numpy().tolist())
        probabilities.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())

    metrics = classification_metrics(y_true, probabilities)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()
    print(f"Using device: {device}")

    train_path = Path(args.data_dir) / "train.csv"
    val_path = Path(args.data_dir) / "val.csv"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run: python scripts/fetch_data.py"
        )

    train_dataset = AntibodyDataset(train_path, max_len=args.max_len)
    val_dataset = AntibodyDataset(val_path, max_len=args.max_len)

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = AntibodyDevelopabilityCNN(
        embedding_dim=args.embedding_dim,
        channels=args.channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    train_labels = train_dataset.df["label"].astype(float).to_numpy()
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)

    if positives > 0 and negatives > 0:
        pos_weight = torch.tensor([negatives / positives], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")
    checkpoint_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    best_val_auprc = -float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, criterion, device, optimizer=None
            )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss={train_metrics['loss']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val AUROC={val_metrics['auroc']:.4f} | "
            f"val AUPRC={val_metrics['auprc']:.4f}"
        )

        score = val_metrics["auprc"]
        if np.isfinite(score) and score > best_val_auprc:
            best_val_auprc = score
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "best_val_metrics": val_metrics,
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / "training_history.csv", index=False)

    save_json(
        {
            "best_validation_auprc": best_val_auprc,
            "epochs_completed": len(history),
            "device": str(device),
            "config": vars(args),
        },
        results_dir / "training_summary.json",
    )

    print("\nTraining complete.")
    print("Best checkpoint: checkpoints/best_model.pt")
    print("Next step: python evaluate.py")


if __name__ == "__main__":
    main()
