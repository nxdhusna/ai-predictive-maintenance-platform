from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.data_loader import PROCESSED_DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "models"
    / "failure_classifier.joblib"
)

TEST_FEATURE_PATH = (
    PROCESSED_DATA_DIR
    / "test_FD001_features.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "failure_classification"
)

OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "official_test_failure_probabilities.csv"
)


def load_classifier_bundle():
    """
    Load the trained classifier and calibrator.
    """

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            "Failure classifier was not found. "
            "Run 'python -m src.failure_classifier' first."
        )

    return joblib.load(
        CLASSIFIER_PATH
    )


def load_test_features():
    """
    Load feature-engineered official test data.
    """

    if not TEST_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "Official test features were not found. "
            "Run 'python -m src.feature_engineering' first."
        )

    return pd.read_csv(
        TEST_FEATURE_PATH
    )


def get_final_test_cycles(dataframe):
    """
    Select one final observation per official test engine.
    """

    final_rows = (
        dataframe
        .sort_values(
            ["engine_id", "cycle"]
        )
        .groupby("engine_id")
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    if (
        final_rows["engine_id"].nunique()
        != 100
    ):
        raise ValueError(
            "Expected 100 official test engines."
        )

    return final_rows


def score_failure_probabilities(
    final_cycle_dataframe,
    classifier_bundle,
):
    """
    Generate calibrated failure-within-30-cycle probabilities.
    """

    base_model = classifier_bundle[
        "base_model"
    ]

    calibrator = classifier_bundle[
        "probability_calibrator"
    ]

    feature_columns = classifier_bundle[
        "feature_columns"
    ]

    missing_features = [
        feature
        for feature in feature_columns
        if feature
        not in final_cycle_dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Official test data is missing features: "
            f"{missing_features}"
        )

    raw_probabilities = (
        base_model.predict_proba(
            final_cycle_dataframe[
                feature_columns
            ]
        )[:, 1]
    )

    calibrated_probabilities = (
        calibrator.predict_proba(
            raw_probabilities.reshape(
                -1,
                1,
            )
        )[:, 1]
    )

    calibrated_probabilities = (
        np.clip(
            calibrated_probabilities,
            0,
            1,
        )
    )

    decision_threshold = float(
        classifier_bundle[
            "decision_threshold"
        ]
    )

    result = final_cycle_dataframe[
        [
            "engine_id",
            "cycle",
            "rul",
            "rul_capped",
        ]
    ].copy()

    result = result.rename(
        columns={
            "cycle": "final_observed_cycle",
            "rul": "actual_rul",
            "rul_capped": (
                "actual_rul_capped"
            ),
        }
    )

    result[
        "raw_failure_probability"
    ] = raw_probabilities

    result[
        "failure_probability"
    ] = calibrated_probabilities

    result[
        "failure_probability_percent"
    ] = (
        calibrated_probabilities
        * 100
    ).round(1)

    result[
        "predicted_within_30_cycles"
    ] = (
        calibrated_probabilities
        >= decision_threshold
    ).astype(int)

    result[
        "actual_within_30_cycles"
    ] = (
        result["actual_rul"]
        <= classifier_bundle[
            "failure_horizon"
        ]
    ).astype(int)

    result[
        "prediction_correct"
    ] = (
        result[
            "predicted_within_30_cycles"
        ]
        == result[
            "actual_within_30_cycles"
        ]
    )

    return result


def create_scoring_summary(
    results,
    classifier_bundle,
):
    """
    Summarise official test probability scoring.
    """

    return {
        "assets_scored": int(
            results["engine_id"].nunique()
        ),
        "failure_horizon_cycles": int(
            classifier_bundle[
                "failure_horizon"
            ]
        ),
        "decision_threshold": float(
            classifier_bundle[
                "decision_threshold"
            ]
        ),
        "average_failure_probability": round(
            float(
                results[
                    "failure_probability"
                ].mean()
            ),
            4,
        ),
        "assets_flagged_within_horizon": int(
            results[
                "predicted_within_30_cycles"
            ].sum()
        ),
        "methodology_note": (
            "Probabilities are generated by the calibrated "
            "failure-within-30-cycles classifier."
        ),
    }


def run_official_failure_scoring():
    """
    Score all 100 official test engines.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    classifier_bundle = (
        load_classifier_bundle()
    )

    test_dataframe = (
        load_test_features()
    )

    final_cycle_dataframe = (
        get_final_test_cycles(
            test_dataframe
        )
    )

    results = score_failure_probabilities(
        final_cycle_dataframe,
        classifier_bundle,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary = create_scoring_summary(
        results,
        classifier_bundle,
    )

    with (
        OUTPUT_DIRECTORY
        / "official_test_failure_summary.json"
    ).open(
        mode="w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            summary,
            summary_file,
            indent=4,
        )

    print(
        "\nOFFICIAL TEST FAILURE PROBABILITIES"
    )

    print("=" * 70)

    print(
        f"Assets scored: "
        f"{summary['assets_scored']}"
    )

    print(
        "Average failure probability: "
        f"{summary['average_failure_probability']:.3f}"
    )

    print(
        "Assets flagged within 30 cycles: "
        f"{summary['assets_flagged_within_horizon']}"
    )

    print(
        f"\nOutput saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    run_official_failure_scoring()