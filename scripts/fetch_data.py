from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tdc.single_pred import Develop


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_pair(value: Any) -> tuple[str, str] | None:
    """Return (heavy, light) if value appears to contain two sequences."""
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
        return str(value[0]), str(value[1])

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("("):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return None
            if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                return str(parsed[0]), str(parsed[1])

    return None


def identify_label_column(df: pd.DataFrame) -> str:
    candidates = [
        col for col in df.columns
        if str(col).lower() in {"y", "label", "target"}
    ]
    if candidates:
        return candidates[0]

    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    # Exclude obvious ID-like numeric columns when possible.
    numeric_cols = [
        col for col in numeric_cols
        if "id" not in str(col).lower()
    ]

    for col in numeric_cols:
        values = set(pd.Series(df[col]).dropna().astype(float).unique().tolist())
        if values.issubset({0.0, 1.0}) and len(values) <= 2:
            return col

    raise ValueError(
        "Could not identify the binary label column. "
        f"Columns were: {list(df.columns)}"
    )


def identify_chain_columns(df: pd.DataFrame, label_col: str):
    lower_map = {str(col).lower(): col for col in df.columns}

    heavy_candidates = [
        "heavy_chain", "heavy", "h_chain", "vh", "sequence_heavy"
    ]
    light_candidates = [
        "light_chain", "light", "l_chain", "vl", "sequence_light"
    ]

    heavy_col = next((lower_map[x] for x in heavy_candidates if x in lower_map), None)
    light_col = next((lower_map[x] for x in light_candidates if x in lower_map), None)

    if heavy_col is not None and light_col is not None:
        return ("separate", heavy_col, light_col)

    for col in df.columns:
        if col == label_col:
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        sample = non_null.iloc[0]
        if parse_pair(sample) is not None:
            return ("paired", col, None)

    raise ValueError(
        "Could not identify heavy/light chain data automatically. "
        f"Columns were: {list(df.columns)}"
    )


def normalize_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    label_col = identify_label_column(df)
    mode, first_col, second_col = identify_chain_columns(df, label_col)

    rows = []
    for row_index, row in df.iterrows():
        if mode == "separate":
            heavy = str(row[first_col])
            light = str(row[second_col])
        else:
            pair = parse_pair(row[first_col])
            if pair is None:
                raise ValueError(
                    f"Could not parse paired chains in {split_name}, row {row_index}."
                )
            heavy, light = pair

        label = float(row[label_col])

        if not heavy or heavy.lower() == "nan":
            raise ValueError(f"Missing heavy chain in {split_name}, row {row_index}.")
        if not light or light.lower() == "nan":
            raise ValueError(f"Missing light chain in {split_name}, row {row_index}.")

        rows.append(
            {
                "antibody_id": f"{split_name}_{row_index}",
                "heavy_chain": heavy.strip().upper(),
                "light_chain": light.strip().upper(),
                "label": label,
            }
        )

    out = pd.DataFrame(rows)

    unique_labels = sorted(out["label"].dropna().unique().tolist())
    if not set(unique_labels).issubset({0.0, 1.0}):
        raise ValueError(
            "Expected a binary 0/1 label for SAbDab_Chen, but found "
            f"{unique_labels[:20]}"
        )

    return out


def summarize(splits: dict[str, pd.DataFrame]) -> dict:
    result = {}
    for name, df in splits.items():
        result[name] = {
            "n_examples": int(len(df)),
            "positive_fraction": float(df["label"].mean()),
            "heavy_length_min": int(df["heavy_chain"].str.len().min()),
            "heavy_length_median": float(df["heavy_chain"].str.len().median()),
            "heavy_length_max": int(df["heavy_chain"].str.len().max()),
            "light_length_min": int(df["light_chain"].str.len().min()),
            "light_length_median": float(df["light_chain"].str.len().median()),
            "light_length_max": int(df["light_chain"].str.len().max()),
        }
    return result


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading SAbDab_Chen from Therapeutics Data Commons...")
    data = Develop(name="SAbDab_Chen")
    raw_splits = data.get_split()

    normalized = {}
    for split_name in ("train", "val", "test"):
        if split_name not in raw_splits:
            raise KeyError(
                f"TDC did not return split '{split_name}'. "
                f"Available keys: {list(raw_splits.keys())}"
            )

        df = raw_splits[split_name].copy()
        print(f"\nRaw {split_name} columns: {list(df.columns)}")
        normalized[split_name] = normalize_split(df, split_name)
        out_path = PROCESSED_DIR / f"{split_name}.csv"
        normalized[split_name].to_csv(out_path, index=False)
        print(f"Saved {len(normalized[split_name])} rows -> {out_path}")

    summary = summarize(normalized)
    summary_path = PROCESSED_DIR / "data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nData summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()
