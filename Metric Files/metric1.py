import os
import pandas as pd
import numpy as np
from math import e


def compute_exponential_time_mape(
    submission,
    ground_truth,
    forecast_horizon=37,
):
    # Deal with date type
    submission['transition_date'] = pd.to_datetime(submission['transition_date'])
    ground_truth["transition_date"] = pd.to_datetime(ground_truth["transition_date"])

    # Rename columns
    ground_truth.rename(
        columns={"transition_date": "real_date", "is_transition": "real_transition"}, inplace=True
    )

    df = ground_truth[["cluster_id", "real_date", "real_transition"]].merge(
        submission[["cluster_id", "transition_date", "is_transition"]], on="cluster_id", how="left"
    )

    # Compute time-error
    time_error = abs(np.round((df["real_date"] - df["transition_date"]) / np.timedelta64(1, "M")))

    df["error"] = time_error / forecast_horizon

    df["exp_error"] = e ** (df["error"])

    # normalize with min max
    df["norm_error"] = (df["exp_error"] - e ** (0)) / (e ** (1) - e ** (0))

    # penalize wrong is_transition value
    df.loc[df["is_transition"] != df["real_transition"], "norm_error"] = 1.0

    # If length of unique transition dates is lower than 1, provide 100% errors
    if len(np.unique(df['transition_date'])) == 1:
        return 1

    return np.mean(df["norm_error"])
