from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data_loader import PROCESSED_DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "rul_model.joblib"
)

MODEL_FEATURE_PATH = (
    PROJECT_ROOT
    / "models"
    / "rul_model_features.txt"
)

MODEL_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "rul_model_metadata.json"
)

TEST_FEATURE_PATH = (
    PROCESSED_DATA_DIR
    / "test_FD001_features.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "test_evaluation"
)

RUL_CAP = 125


def prepare_output_directory():
    """
    Create the official test-evaluation output directory.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_model():
    """
    Load the trained RUL prediction model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained RUL model was not found. "
            "Run 'python -m src.model_training' first."
        )

    return joblib.load(MODEL_PATH)


def load_model_features():
    """
    Load the exact ordered feature list used during training.
    """

    if not MODEL_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "The saved model-feature list was not found."
        )

    with MODEL_FEATURE_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as feature_file:
        feature_columns = [
            line.strip()
            for line in feature_file
            if line.strip()
        ]

    if not feature_columns:
        raise ValueError(
            "The saved model-feature list is empty."
        )

    return feature_columns


def load_test_feature_dataframe():
    """
    Load the feature-engineered FD001 test data.
    """

    if not TEST_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "The test feature dataset was not found. "
            "Run 'python -m src.feature_engineering' first."
        )

    dataframe = pd.read_csv(
        TEST_FEATURE_PATH
    )

    required_columns = {
        "engine_id",
        "cycle",
        "rul",
        "rul_capped",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Required test columns are missing: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def get_final_engine_cycles(
    test_dataframe,
    expected_engine_count=100,
):
    """
    Select the final observed cycle for each test engine.

    This produces one prediction row per engine, matching the
    official C-MAPSS RUL evaluation format.
    """

    final_cycle_dataframe = (
        test_dataframe
        .sort_values(
            ["engine_id", "cycle"]
        )
        .groupby(
            "engine_id",
            as_index=False,
        )
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    engine_count = (
        final_cycle_dataframe[
            "engine_id"
        ].nunique()
    )

    if (
            expected_engine_count is not None
            and engine_count != expected_engine_count
    ):
        raise ValueError(
            f"Expected {expected_engine_count} test engines, "
            f"but found {engine_count}."
        )

    return final_cycle_dataframe

def validate_model_features(
    dataframe,
    feature_columns,
):
    """
    Confirm that every feature required by the trained model
    exists in the official test dataframe.
    """

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in dataframe.columns
    ]

    prohibited_features = {
        "engine_id",
        "rul",
        "rul_capped",
        "observed_life_fraction",
    }

    leaked_features = [
        feature
        for feature in feature_columns
        if feature in prohibited_features
    ]

    if missing_features:
        raise ValueError(
            "Test data is missing model features: "
            f"{missing_features}"
        )

    if leaked_features:
        raise ValueError(
            "Potentially leaked features were found: "
            f"{leaked_features}"
        )


def calculate_metrics(
    actual_values,
    predicted_values,
):
    """
    Calculate standard regression metrics.
    """

    mae = mean_absolute_error(
        actual_values,
        predicted_values,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values,
        )
    )

    r_squared = r2_score(
        actual_values,
        predicted_values,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r_squared),
    }


def generate_test_predictions(
    model,
    final_cycle_dataframe,
    feature_columns,
):
    """
    Generate one RUL prediction for each official test engine.
    """

    model_inputs = final_cycle_dataframe[
        feature_columns
    ]

    predicted_rul = model.predict(
        model_inputs
    )

    predicted_rul = np.clip(
        predicted_rul,
        0,
        RUL_CAP,
    )

    results = final_cycle_dataframe[
        [
            "engine_id",
            "cycle",
            "rul",
            "rul_capped",
        ]
    ].copy()

    results = results.rename(
        columns={
            "cycle": "final_observed_cycle",
            "rul": "actual_rul",
            "rul_capped": "actual_rul_capped",
        }
    )

    results["predicted_rul"] = predicted_rul

    results["residual_capped"] = (
        results["actual_rul_capped"]
        - results["predicted_rul"]
    )

    results["absolute_error_capped"] = (
        results["residual_capped"].abs()
    )

    results["residual_raw"] = (
        results["actual_rul"]
        - results["predicted_rul"]
    )

    results["absolute_error_raw"] = (
        results["residual_raw"].abs()
    )

    return results


def calculate_evaluation_summary(
    results,
):
    """
    Calculate both capped-target and raw-target metrics.

    The model was trained on capped RUL, so capped metrics are
    the primary model-comparison results. Raw metrics are included
    for transparency against the original test targets.
    """

    capped_metrics = calculate_metrics(
        results["actual_rul_capped"],
        results["predicted_rul"],
    )

    raw_metrics = calculate_metrics(
        results["actual_rul"],
        results["predicted_rul"],
    )

    summary = {
        "number_of_test_engines": int(
            results["engine_id"].nunique()
        ),
        "rul_cap": RUL_CAP,
        "primary_evaluation": "capped_rul",
        "capped_rul_metrics": capped_metrics,
        "raw_rul_metrics": raw_metrics,
        "mean_predicted_rul": float(
            results["predicted_rul"].mean()
        ),
        "mean_actual_capped_rul": float(
            results["actual_rul_capped"].mean()
        ),
        "maximum_absolute_capped_error": float(
            results["absolute_error_capped"].max()
        ),
        "median_absolute_capped_error": float(
            results["absolute_error_capped"].median()
        ),
        "limitations": [
            (
                "The model was trained using a capped RUL "
                "target of 125 cycles."
            ),
            (
                "Capped metrics are the primary fair comparison "
                "for the current trained model."
            ),
            (
                "The dataset is simulated and is not validated "
                "for real safety-critical maintenance use."
            ),
        ],
    }

    return summary


def plot_actual_vs_predicted(
    results,
    metrics,
):
    """
    Plot official capped RUL against predicted RUL.
    """

    plt.figure(figsize=(8, 7))

    plt.scatter(
        results["actual_rul_capped"],
        results["predicted_rul"],
        alpha=0.70,
        s=45,
        edgecolor="black",
        linewidth=0.3,
    )

    plt.plot(
        [0, RUL_CAP],
        [0, RUL_CAP],
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Perfect prediction",
    )

    plt.title(
        "Official FD001 Test Set: Actual vs Predicted RUL"
    )

    plt.xlabel("Actual Capped RUL")
    plt.ylabel("Predicted Capped RUL")

    metric_text = (
        f"MAE: {metrics['mae']:.2f}\n"
        f"RMSE: {metrics['rmse']:.2f}\n"
        f"R²: {metrics['r2']:.3f}"
    )

    plt.text(
        0.05,
        0.95,
        metric_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    plt.legend()
    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "official_test_actual_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_residuals(
    results,
):
    """
    Plot official test residual distribution.
    """

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=results,
        x="residual_capped",
        bins=20,
        kde=True,
    )

    plt.axvline(
        x=0,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    plt.title(
        "Official FD001 Test Prediction Residuals"
    )

    plt.xlabel(
        "Residual: Actual Capped RUL - Predicted RUL"
    )

    plt.ylabel("Number of Engines")
    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "official_test_residuals.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_worst_predictions(
    results,
    number_of_engines=15,
):
    """
    Save and visualise engines with the largest absolute error.
    """

    worst_predictions = (
        results.nlargest(
            number_of_engines,
            "absolute_error_capped",
        )
        .sort_values(
            "absolute_error_capped",
            ascending=True,
        )
        .copy()
    )

    csv_output_path = (
        OUTPUT_DIRECTORY
        / "worst_test_predictions.csv"
    )

    worst_predictions.to_csv(
        csv_output_path,
        index=False,
    )

    plt.figure(figsize=(11, 7))

    sns.barplot(
        data=worst_predictions,
        x="absolute_error_capped",
        y=worst_predictions[
            "engine_id"
        ].astype(str),
        orient="h",
    )

    plt.title(
        "Official Test Engines with Largest RUL Errors"
    )

    plt.xlabel("Absolute Error in Capped RUL Cycles")
    plt.ylabel("Engine ID")
    plt.tight_layout()

    image_output_path = (
        OUTPUT_DIRECTORY
        / "worst_test_predictions.png"
    )

    plt.savefig(
        image_output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_model_comparison_metadata(
    evaluation_summary,
):
    """
    Combine official-test metrics with the saved model metadata.
    """

    combined_metadata = {
        "official_test_evaluation": evaluation_summary,
    }

    if MODEL_METADATA_PATH.exists():
        with MODEL_METADATA_PATH.open(
            mode="r",
            encoding="utf-8",
        ) as metadata_file:
            validation_metadata = json.load(
                metadata_file
            )

        combined_metadata[
            "grouped_validation_evaluation"
        ] = validation_metadata

    output_path = (
        OUTPUT_DIRECTORY
        / "official_test_metrics.json"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            combined_metadata,
            output_file,
            indent=4,
        )


def run_official_test_evaluation():
    """
    Run the complete official FD001 endpoint evaluation.
    """

    prepare_output_directory()

    model = load_model()
    feature_columns = load_model_features()

    test_dataframe = (
        load_test_feature_dataframe()
    )

    final_cycle_dataframe = (
        get_final_engine_cycles(
            test_dataframe
        )
    )

    validate_model_features(
        final_cycle_dataframe,
        feature_columns,
    )

    results = generate_test_predictions(
        model,
        final_cycle_dataframe,
        feature_columns,
    )

    evaluation_summary = (
        calculate_evaluation_summary(
            results
        )
    )

    capped_metrics = evaluation_summary[
        "capped_rul_metrics"
    ]

    results.to_csv(
        OUTPUT_DIRECTORY
        / "official_test_predictions.csv",
        index=False,
    )

    plot_actual_vs_predicted(
        results,
        capped_metrics,
    )

    plot_residuals(
        results
    )

    save_worst_predictions(
        results
    )

    save_model_comparison_metadata(
        evaluation_summary
    )

    print("\nOFFICIAL FD001 TEST EVALUATION")
    print("=" * 70)

    print(
        f"Test engines evaluated: "
        f"{evaluation_summary['number_of_test_engines']}"
    )

    print("\nPRIMARY CAPPED-RUL METRICS")
    print("-" * 70)

    print(
        f"MAE:  "
        f"{capped_metrics['mae']:.3f} cycles"
    )

    print(
        f"RMSE: "
        f"{capped_metrics['rmse']:.3f} cycles"
    )

    print(
        f"R²:   "
        f"{capped_metrics['r2']:.3f}"
    )

    raw_metrics = evaluation_summary[
        "raw_rul_metrics"
    ]

    print("\nRAW-RUL TRANSPARENCY METRICS")
    print("-" * 70)

    print(
        f"MAE:  "
        f"{raw_metrics['mae']:.3f} cycles"
    )

    print(
        f"RMSE: "
        f"{raw_metrics['rmse']:.3f} cycles"
    )

    print(
        f"R²:   "
        f"{raw_metrics['r2']:.3f}"
    )

    print("\nERROR DISTRIBUTION")
    print("-" * 70)

    print(
        "Median absolute capped error: "
        f"{evaluation_summary['median_absolute_capped_error']:.3f}"
    )

    print(
        "Maximum absolute capped error: "
        f"{evaluation_summary['maximum_absolute_capped_error']:.3f}"
    )

    print(
        "\nOutputs saved to:"
    )

    print(OUTPUT_DIRECTORY)


if __name__ == "__main__":
    run_official_test_evaluation()