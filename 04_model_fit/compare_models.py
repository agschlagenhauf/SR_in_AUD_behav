#!/usr/bin/env python3
"""
Model comparison via AIC and BIC.

This script takes the aggregated parameter-estimate CSVs produced by
`run_estimation.py` (one file per agent type, one row per subject)
and computes AIC and BIC for each subject–model pair.

Expected input files (created by run_estimation.py):
    results/best_parameters_{method}_{agent_type}.csv

Each file should contain at least:
    - subject
    - agent_type
    - method
    - alpha_rwq
    - alpha_mt
    - beta
    - gamma
    - best_likelihood

Optionally, you can also supply the behavioral data file to compute
the exact number of trials per subject. If not provided, the script
will still compute AIC (which does not depend on n) and will set
`n_choices` and `BIC` to NaN.

Usage
-----

python compare_models.py \\
    --results-dir results_real_data \\
    --output results_real_data/model_comparison_lbfgsb.csv

"""

import argparse
import os
import glob
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare models using AIC and BIC.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results_real_data",
        help="Directory containing best_parameters_{method}_{agent_type}.csv files "
             "(default: results)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output CSV path for the combined model comparison table. "
             "Default: results/model_comparison_{method}.csv",
    )
    return parser.parse_args()


def infer_k_from_params(row: pd.Series) -> int:
    """
    Manually set number of free parameters k per model (agent_type).

    Adjust the mapping below if you change which parameters are fitted
    for a given model class.
    """
    agent_type = str(row.get("agent_type"))

    # Manual mapping: agent_type -> k (number of free parameters)
    k_by_model = {
        # 4-parameter SR-like agents
        "sr": 2,
        "redsr_2": 2,
        "redsr_3": 2,
        "redsr_4": 2,
        "randsr_noupdate": 2,
        "randsr_wupdate": 2,
        "redsr_2_randsr_wupdate": 2,
        "redsr_3_randsr_wupdate": 2,
        "redsr_4_randsr_wupdate": 2,
        "hybrid_mf_redsr_4_randsr_wupdate": 3,
        "hybrid_mf_randsr_noupdate": 3,
        "hybrid_mf_randsr_wupdate": 3,
        "hybrid_mf_mb_learnt": 4,
        "mb": 2,
        "mb_learnt": 3,
        "mf": 2,
    }

    if agent_type not in k_by_model:
        raise ValueError(
            f"Unknown agent_type '{agent_type}' in infer_k_from_params; "
            f"please add it to k_by_model."
        )

    return k_by_model[agent_type]


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir
    method = "lbfgsb"

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Pattern: results/best_parameters_{method}_{agent_type}.csv
    pattern = os.path.join(results_dir, f"best_parameters_{method}_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No aggregated parameter files found matching pattern: {pattern}"
        )

    # ------------------------------------------------------------------
    # Model fit per participant
    # ------------------------------------------------------------------

    all_rows = []

    for path in files:

        df = pd.read_csv(path)

        for _, row in df.iterrows():
            subj = str(row["subject"])
            ll = float(row["best_likelihood"])

            k = infer_k_from_params(row)
            aic = 2 * k - 2 * ll

            n = 250 # set number of choices per subject that contribute to log likelihood (5 conditions * 25 free-choice learning and test trials * 2 choices per trial)
            bic = k * math.log(n) - 2 * ll
        
            out_row = dict(row)  # copy original columns
            out_row["k_params"] = k
            out_row["n_choices"] = n
            out_row["AIC"] = aic
            out_row["BIC"] = bic
            all_rows.append(out_row)

    comparison_df = pd.DataFrame(all_rows)

    # Determine output path
    if args.output is not None:
        output_path = args.output
    else:
        output_path = os.path.join(results_dir, f"model_fit_per_participant_{method}.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison_df.to_csv(output_path, index=False)

    print(f"📁 Model comparison table written to: {output_path}")


    # ------------------------------------------------------------------
    # Model comparison summary
    # ------------------------------------------------------------------
    summary_rows = []
    for agent_type, group in comparison_df.groupby("agent_type"):
        # Identify finite vs. infinite likelihoods
        is_finite_ll = np.isfinite(group["best_likelihood"].values)
        n_total = int(len(group))
        n_excluded = int((~is_finite_ll).sum())
        n_used = int(is_finite_ll.sum())

        # Print exclusion info per model
        print(
            f"\nModel {agent_type}: {n_excluded} subjects with infinite likelihood "
            f"(used {n_used} of {n_total} for aggregates)."
        )

        valid_group = group[is_finite_ll]

        # k should be constant within a model; take the first non-null value
        if n_used > 0:
            k_vals = valid_group["k_params"].dropna().unique()
        else:
            k_vals = []
        k = int(k_vals[0]) if len(k_vals) > 0 else None

        # n is fixed if BIC was computed, otherwise NaN
        if (
            n_used > 0
            and "n_choices" in valid_group.columns
            and valid_group["n_choices"].notna().any()
        ):
            n_vals = valid_group["n_choices"].dropna().unique()
            n = int(n_vals[0]) if len(n_vals) > 0 else None
        else:
            n = None

        if n_used > 0:
            ll_mean = valid_group["best_likelihood"].mean()
            ll_sd = valid_group["best_likelihood"].std(ddof=1)

            aic_mean = valid_group["AIC"].mean()
            aic_sd = valid_group["AIC"].std(ddof=1)
            bic_mean = valid_group["BIC"].mean() if "BIC" in valid_group.columns else float("nan")
            bic_sd = valid_group["BIC"].std(ddof=1) if "BIC" in valid_group.columns else float("nan")
        else:
            ll_mean = float("nan")
            ll_sd = float("nan")
            aic_mean = float("nan")
            aic_sd = float("nan")
            bic_mean = float("nan")
            bic_sd = float("nan")

        summary_rows.append(
            {
                "agent_type": agent_type,
                "n_subjects_total": n_total,
                "n_subjects_excluded_inf": n_excluded,
                "n_subjects_used": n_used,
                "k_params": k,
                "n_choices": n,
                "loglik_mean": ll_mean,
                "loglik_sd": ll_sd,
                "AIC_mean": aic_mean,
                "AIC_sd": aic_sd,
                "BIC_mean": bic_mean,
                "BIC_sd": bic_sd,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("AIC_mean")

    # Print summary table to console
    print("\n📊 Model-level summary:")
    print(summary_df.to_string(index=False))

    # Save summary table as CSV next to the detailed comparison file
    summary_path = os.path.join(results_dir, f"model_comparison_summary_{method}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n📁 Model summary table written to: {summary_path}")

    # ------------------------------------------------------------------
    # Parameter summary
    # ------------------------------------------------------------------
    param_cols = [col for col in ["alpha_rwq", "alpha_mt", "beta", "gamma", "w"] if col in comparison_df.columns]

    param_summary_rows = []
    for agent_type, group in comparison_df.groupby("agent_type"):
        row: Dict[str, float] = {"agent_type": agent_type}
        for p in param_cols:
            row[f"{p}_mean"] = group[p].mean()
            row[f"{p}_sd"] = group[p].std(ddof=1)
        param_summary_rows.append(row)

    param_summary_df = pd.DataFrame(param_summary_rows).sort_values("agent_type")

    # Print parameter summary to console
    print("\n📊 Parameter summary (per model):")
    print(param_summary_df.to_string(index=False))

    param_summary_path = os.path.join(results_dir, f"parameter_summary_{method}.csv")
    param_summary_df.to_csv(param_summary_path, index=False)
    print(f"\n📁 Parameter summary table written to: {param_summary_path}")


if __name__ == "__main__":
    main()

