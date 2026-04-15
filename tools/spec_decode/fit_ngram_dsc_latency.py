#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


_REALIZED_PREFIX = "NGRAM_DSC_REALIZED "
_KEY_VALUE_PATTERN = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


@dataclass
class RealizedRecord:
    effective_num_spec_tokens: int
    decode_token_load: int
    smoothed_total_num_scheduled_tokens: float
    predicted_generated_tokens: float
    realized_generated_tokens: float
    predicted_latency_s: float
    realized_latency_s: float
    predicted_goodput: float
    realized_goodput: float


def _parse_key_values(line: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in _KEY_VALUE_PATTERN.finditer(line)}


def parse_realized_records(log_path: Path) -> list[RealizedRecord]:
    records: list[RealizedRecord] = []
    with log_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            if _REALIZED_PREFIX not in raw_line:
                continue
            fields = _parse_key_values(raw_line)
            required_keys = {
                "effective_num_spec_tokens",
                "decode_token_load",
                "smoothed_total_num_scheduled_tokens",
                "predicted_generated_tokens",
                "realized_generated_tokens",
                "predicted_latency_s",
                "realized_latency_s",
                "predicted_goodput",
                "realized_goodput",
            }
            if not required_keys.issubset(fields):
                continue
            records.append(
                RealizedRecord(
                    effective_num_spec_tokens=int(fields["effective_num_spec_tokens"]),
                    decode_token_load=int(fields["decode_token_load"]),
                    smoothed_total_num_scheduled_tokens=float(
                        fields["smoothed_total_num_scheduled_tokens"]
                    ),
                    predicted_generated_tokens=float(
                        fields["predicted_generated_tokens"]
                    ),
                    realized_generated_tokens=float(fields["realized_generated_tokens"]),
                    predicted_latency_s=float(fields["predicted_latency_s"]),
                    realized_latency_s=float(fields["realized_latency_s"]),
                    predicted_goodput=float(fields["predicted_goodput"]),
                    realized_goodput=float(fields["realized_goodput"]),
                )
            )
    return records


def _drop_outliers(
    records: list[RealizedRecord],
    *,
    drop_first_n: int,
    max_realized_latency_s: float | None,
    max_latency_ratio_to_median: float | None,
) -> list[RealizedRecord]:
    trimmed = records[drop_first_n:]
    if not trimmed:
        return []
    if max_realized_latency_s is not None:
        trimmed = [
            record
            for record in trimmed
            if record.realized_latency_s <= max_realized_latency_s
        ]
    if max_latency_ratio_to_median is not None and trimmed:
        median_latency = float(np.median([r.realized_latency_s for r in trimmed]))
        latency_limit = median_latency * max_latency_ratio_to_median
        trimmed = [
            record for record in trimmed if record.realized_latency_s <= latency_limit
        ]
    return trimmed


def _design_matrix(records: list[RealizedRecord]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [
            [
                1.0,
                float(record.decode_token_load),
                record.smoothed_total_num_scheduled_tokens,
                float(record.effective_num_spec_tokens),
                float(record.effective_num_spec_tokens)
                * record.smoothed_total_num_scheduled_tokens,
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    y = np.array([record.realized_latency_s for record in records], dtype=np.float64)
    return x, y


def _fit_nonnegative_least_squares(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    active = list(range(x.shape[1]))
    coeffs = np.zeros(x.shape[1], dtype=np.float64)

    while active:
        active_x = x[:, active]
        active_coeffs, *_ = np.linalg.lstsq(active_x, y, rcond=None)
        negative_indices = [i for i, value in enumerate(active_coeffs) if value < 0.0]
        if not negative_indices:
            coeffs[np.array(active)] = active_coeffs
            break
        most_negative = min(negative_indices, key=lambda idx: active_coeffs[idx])
        del active[most_negative]

    predictions = x @ coeffs
    return coeffs, predictions


def _fit_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    residual = y_true - y_pred
    rmse = math.sqrt(float(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    total_variance = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = None if total_variance == 0.0 else float(
        1.0 - (np.sum(residual**2) / total_variance)
    )
    return {"rmse_s": rmse, "mae_s": mae, "r2": r2}


def build_output_payload(
    *,
    log_path: Path,
    original_num_samples: int,
    records: list[RealizedRecord],
    coefficients: np.ndarray,
    metrics: dict[str, float | None],
) -> dict[str, object]:
    return {
        "latency_model": "profiled",
        "fitted_from": str(log_path),
        "num_samples": len(records),
        "dropped_samples": original_num_samples - len(records),
        "feature_columns": [
            "intercept",
            "decode_token_load",
            "smoothed_total_num_scheduled_tokens",
            "effective_num_spec_tokens",
            "effective_num_spec_tokens_x_smoothed_total_num_scheduled_tokens",
        ],
        "fit_metrics": metrics,
        "coefficients": {
            "intercept_s": float(coefficients[0]),
            "decode_token_load_coeff_s": float(coefficients[1]),
            "scheduled_tokens_coeff_s": float(coefficients[2]),
            "spec_tokens_coeff_s": float(coefficients[3]),
            "spec_scheduled_tokens_interaction_coeff_s": float(coefficients[4]),
        },
        "example_speculative_config_overrides": {
            "ngram_dsc_latency_model": "profiled",
            "ngram_dsc_profiled_latency_coefficients_path": "./ngram_dsc_latency_coeffs.json",
        },
        "records_preview": [asdict(record) for record in records[:5]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit profiled TurboSpec latency coefficients from NGRAM_DSC_REALIZED logs."
    )
    parser.add_argument("--log", required=True, help="Path to dsc.log")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write fitted coefficient JSON",
    )
    parser.add_argument(
        "--drop-first-n",
        type=int,
        default=1,
        help="Drop the first N realized samples to avoid warmup skew. Default: 1.",
    )
    parser.add_argument(
        "--max-realized-latency-s",
        type=float,
        default=None,
        help="Optional hard cutoff for realized latency.",
    )
    parser.add_argument(
        "--max-latency-ratio-to-median",
        type=float,
        default=3.0,
        help="Discard realized-latency outliers above median * ratio. Default: 3.0.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    output_path = Path(args.output)

    records = parse_realized_records(log_path)
    original_num_samples = len(records)
    if not records:
        raise SystemExit(
            "No usable NGRAM_DSC_REALIZED samples found. "
            "Rerun with logs that include decode_token_load and "
            "smoothed_total_num_scheduled_tokens."
        )

    records = _drop_outliers(
        records,
        drop_first_n=max(0, args.drop_first_n),
        max_realized_latency_s=args.max_realized_latency_s,
        max_latency_ratio_to_median=args.max_latency_ratio_to_median,
    )
    if not records:
        raise SystemExit("All samples were filtered out; adjust the filtering flags.")

    x, y = _design_matrix(records)
    coefficients, predictions = _fit_nonnegative_least_squares(x, y)
    payload = build_output_payload(
        log_path=log_path,
        original_num_samples=original_num_samples,
        records=records,
        coefficients=coefficients,
        metrics=_fit_metrics(y, predictions),
    )

    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["coefficients"], indent=2))
    print(f"Wrote fitted coefficients to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
