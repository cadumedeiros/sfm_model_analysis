"""Compara numericamente duas tabelas mestre pelo Simulation_ID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from master_table import normalize_simulation_id


IGNORED_NUMERIC_COLUMNS = {
    "processing_seconds",
}


def load_master_table(path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        frame = pd.read_excel(path, sheet_name="Master_Table")
    else:
        raise ValueError("A tabela mestre deve ser CSV ou Excel.")
    if "simulation_id" not in frame.columns:
        raise ValueError(
            f"{path.name} não possui a coluna simulation_id."
        )
    frame = frame.copy()
    frame["_join_id"] = frame["simulation_id"].map(normalize_simulation_id)
    if frame["_join_id"].duplicated().any():
        raise ValueError(
            f"{path.name} possui simulation_id duplicados."
        )
    return frame


def compare_master_tables(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[pd.DataFrame, dict]:
    reference_ids = set(reference["_join_id"])
    candidate_ids = set(candidate["_join_id"])
    common_ids = sorted(reference_ids.intersection(candidate_ids))
    if not common_ids:
        raise ValueError("As tabelas não possuem modelos em comum.")

    left = reference.set_index("_join_id").loc[common_ids]
    right = candidate.set_index("_join_id").loc[common_ids]
    common_columns = sorted(
        set(left.columns).intersection(right.columns)
        - {"simulation_id"}
        - IGNORED_NUMERIC_COLUMNS
    )
    numeric_columns = []
    for column in common_columns:
        left_numeric = pd.to_numeric(left[column], errors="coerce")
        right_numeric = pd.to_numeric(right[column], errors="coerce")
        if left_numeric.notna().any() or right_numeric.notna().any():
            numeric_columns.append(column)

    rows = []
    for column in numeric_columns:
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(float)
        both_missing = np.isnan(a) & np.isnan(b)
        equal = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        valid_difference = ~(both_missing | np.isnan(a) | np.isnan(b))
        absolute = np.full(a.shape, np.nan, dtype=float)
        absolute[valid_difference] = np.abs(
            a[valid_difference] - b[valid_difference]
        )
        denominator = np.maximum(np.abs(a), atol)
        relative = np.full(a.shape, np.nan, dtype=float)
        relative[valid_difference] = (
            absolute[valid_difference]
            / denominator[valid_difference]
        )
        rows.append(
            {
                "column": column,
                "models_compared": len(common_ids),
                "mismatches": int((~equal).sum()),
                "max_absolute_difference": (
                    float(np.nanmax(absolute))
                    if np.isfinite(absolute).any()
                    else 0.0
                ),
                "max_relative_difference": (
                    float(np.nanmax(relative))
                    if np.isfinite(relative).any()
                    else 0.0
                ),
            }
        )
    details = pd.DataFrame(rows).sort_values(
        ["mismatches", "max_absolute_difference"],
        ascending=[False, False],
    )
    summary = {
        "reference_models": len(reference),
        "candidate_models": len(candidate),
        "common_models": len(common_ids),
        "missing_in_candidate": sorted(
            reference_ids.difference(candidate_ids)
        ),
        "extra_in_candidate": sorted(
            candidate_ids.difference(reference_ids)
        ),
        "numeric_columns_compared": len(details),
        "columns_with_mismatches": int(
            (details["mismatches"] > 0).sum()
        ),
        "rtol": rtol,
        "atol": atol,
    }
    return details, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara uma extração de referência da interface com a "
            "extração em lote."
        )
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default="comparison_report.csv")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    reference = load_master_table(args.reference)
    candidate = load_master_table(args.candidate)
    details, summary = compare_master_tables(
        reference,
        candidate,
        rtol=args.rtol,
        atol=args.atol,
    )
    output = Path(args.output).expanduser().resolve()
    details.to_csv(output, index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Relatório: {output}")
    return 0 if summary["columns_with_mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
