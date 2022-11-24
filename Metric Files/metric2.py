import os
import pandas as pd
import numpy as np
from math import e

def penalized_mape_error(
    submission_one: pd.DataFrame,
    submission_two: pd.DataFrame,
    ground_truth: pd.DataFrame,
    benchmark: pd.DataFrame,
    penalization_factor: int = 1,
):
    # Deal with date type
    submission_two['date'] = pd.to_datetime(submission_two['date'])
    submission_one['transition_date'] = pd.to_datetime(submission_one['transition_date'])
    benchmark['date'] = pd.to_datetime(benchmark['date'])
    ground_truth["date"] = pd.to_datetime(ground_truth["date"])

    benchmark.rename(columns={"forecast": "benchmark_fcst"}, inplace=True)

    # Merge to know the transition_date
    submission_two = submission_two.merge(submission_one, on="cluster_id", how="left")
    submission_two = submission_two.merge(benchmark, on=["cluster_id", "date"], how="left")

    # Check those conditions in which a penalization is needed
    cond_one = submission_two["date"] < submission_two["transition_date"]
    cond_two = submission_two["benchmark_fcst"] != submission_two["forecast"]
    cond_three = submission_two["is_transition"] == "YES"
    submission_two["penalization"] = np.where(cond_one & cond_two & cond_three, "YES", "NO")

    # Rename columns
    ground_truth.rename(columns={"volume": "real_actuals"}, inplace=True)

    df = ground_truth[["cluster_id", "date", "real_actuals"]].merge(
        submission_two[["cluster_id", "date", "forecast","penalization"]],
        on=["cluster_id", "date"],
        how="left",
    )

    # Compute ape per cluster-id-date
    df["error_method"] = abs(df["forecast"] - df["real_actuals"]) / df["real_actuals"]

    df["error_method"] = np.where(df["penalization"] == "YES", penalization_factor, df["error_method"])

    return np.mean(df["error_method"])
