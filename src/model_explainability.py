from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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

TEST_FEATURE_PATH = (
    PROCESSED_DATA_DIR
    / "test_FD001_features.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "explainability"
)

GLOBAL_SAMPLE_SIZE = 1000
RANDOM_STATE = 42
TOP_GLOBAL_FEATURES = 20
TOP_LOCAL_FEATURES = 12


def prepare_output_directory():
    """
    Create the SHAP explainability output directory.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_rul_model():
    """
    Load the trained Remaining Useful Life model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained RUL model was not found. "
            "Run 'python -m src.model_training' first."
        )

    return joblib.load(MODEL_PATH)


def load_model_features():
    """
    Load the exact ordered feature list used by the RUL model.
    """

    if not MODEL_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "The RUL model feature list was not found."
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
            "The RUL model feature list is empty."
        )

    return feature_columns


def load_test_feature_data():
    """
    Load the feature-engineered FD001 test dataset.
    """

    if not TEST_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "The feature-engineered test dataset was not found. "
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
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Test feature data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def validate_model_features(
    dataframe,
    feature_columns,
):
    """
    Confirm that all model features exist and no target leakage
    fields are included in the explanation input.
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
            "The explanation dataset is missing model features: "
            f"{missing_features}"
        )

    if leaked_features:
        raise ValueError(
            "Potential target-leakage features were found: "
            f"{leaked_features}"
        )


def get_final_cycle_data(
    dataframe,
):
    """
    Select the final observed cycle from each test engine.
    """

    final_cycle_dataframe = (
        dataframe
        .sort_values(
            ["engine_id", "cycle"]
        )
        .groupby("engine_id")
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    engine_count = (
        final_cycle_dataframe[
            "engine_id"
        ].nunique()
    )

    if engine_count != 100:
        raise ValueError(
            "Expected 100 final-cycle test engines, "
            f"but found {engine_count}."
        )

    return final_cycle_dataframe


def create_global_sample(
    dataframe,
):
    """
    Create a reproducible sample for fleet-level SHAP analysis.

    Sampling reduces computation while retaining observations
    across the complete test dataset.
    """

    sample_size = min(
        GLOBAL_SAMPLE_SIZE,
        len(dataframe),
    )

    return (
        dataframe.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )
        .sort_values(
            ["engine_id", "cycle"]
        )
        .reset_index(drop=True)
    )


def create_tree_explainer(
    model,
):
    """
    Create a SHAP TreeExplainer for the fitted tree ensemble.
    """

    return shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )


def normalise_shap_values(
    shap_values,
):
    """
    Convert SHAP output into a two-dimensional NumPy array.

    This protects the pipeline against minor SHAP output-format
    differences across supported versions.
    """

    values = np.asarray(
        shap_values
    )

    if values.ndim == 3:
        values = values[:, :, 0]

    if values.ndim != 2:
        raise ValueError(
            "Unexpected SHAP value dimensions: "
            f"{values.shape}"
        )

    return values


def extract_expected_value(
    explainer,
):
    """
    Convert the SHAP expected value into one numeric value.
    """

    expected_value = np.asarray(
        explainer.expected_value
    ).reshape(-1)

    if expected_value.size == 0:
        raise ValueError(
            "SHAP expected value was empty."
        )

    return float(
        expected_value[0]
    )


def calculate_shap_values(
    explainer,
    feature_dataframe,
):
    """
    Calculate SHAP values for the supplied feature observations.
    """

    shap_values = explainer.shap_values(
        feature_dataframe
    )

    return normalise_shap_values(
        shap_values
    )


def validate_shap_additivity(
    model,
    feature_dataframe,
    shap_values,
    expected_value,
    tolerance=1e-4,
):
    """
    Verify that expected value plus feature contributions
    reconstructs the model prediction.
    """

    model_predictions = model.predict(
        feature_dataframe
    )

    reconstructed_predictions = (
        expected_value
        + shap_values.sum(axis=1)
    )

    maximum_difference = float(
        np.max(
            np.abs(
                model_predictions
                - reconstructed_predictions
            )
        )
    )

    if maximum_difference > tolerance:
        raise ValueError(
            "SHAP additivity validation failed. "
            f"Maximum difference: {maximum_difference}"
        )

    return maximum_difference


def create_global_importance(
    shap_values,
    feature_columns,
):
    """
    Rank features by mean absolute SHAP contribution.
    """

    mean_absolute_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    signed_mean_shap = np.mean(
        shap_values,
        axis=0,
    )

    importance_dataframe = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_absolute_shap": (
                mean_absolute_shap
            ),
            "mean_signed_shap": (
                signed_mean_shap
            ),
        }
    )

    importance_dataframe = (
        importance_dataframe
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_dataframe[
        "importance_rank"
    ] = range(
        1,
        len(importance_dataframe) + 1,
    )

    return importance_dataframe


def create_local_explanations(
    final_cycle_dataframe,
    shap_values,
    feature_columns,
    expected_value,
    model,
):
    """
    Create feature-contribution records for every test engine.
    """

    model_inputs = final_cycle_dataframe[
        feature_columns
    ]

    model_predictions = model.predict(
        model_inputs
    )

    explanation_records = []
    summary_records = []

    for row_position in range(
        len(final_cycle_dataframe)
    ):
        engine_id = int(
            final_cycle_dataframe.iloc[
                row_position
            ]["engine_id"]
        )

        cycle = int(
            final_cycle_dataframe.iloc[
                row_position
            ]["cycle"]
        )

        predicted_rul = float(
            model_predictions[row_position]
        )

        feature_values = (
            model_inputs.iloc[row_position]
        )

        engine_shap_values = (
            shap_values[row_position]
        )

        engine_contributions = []

        for feature_position, feature_name in enumerate(
            feature_columns
        ):
            contribution = float(
                engine_shap_values[
                    feature_position
                ]
            )

            feature_value = float(
                feature_values[
                    feature_name
                ]
            )

            engine_contributions.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "feature": feature_name,
                    "feature_value": feature_value,
                    "shap_value": contribution,
                    "absolute_shap_value": abs(
                        contribution
                    ),
                    "contribution_direction": (
                        "Increases predicted RUL"
                        if contribution > 0
                        else (
                            "Reduces predicted RUL"
                            if contribution < 0
                            else "No material contribution"
                        )
                    ),
                }
            )

        engine_contributions = sorted(
            engine_contributions,
            key=lambda record: (
                record["absolute_shap_value"]
            ),
            reverse=True,
        )

        for contribution_rank, record in enumerate(
            engine_contributions,
            start=1,
        ):
            record[
                "contribution_rank"
            ] = contribution_rank

            explanation_records.append(
                record
            )

        top_negative_features = [
            record["feature"]
            for record in engine_contributions
            if record["shap_value"] < 0
        ][:3]

        top_positive_features = [
            record["feature"]
            for record in engine_contributions
            if record["shap_value"] > 0
        ][:3]

        summary_records.append(
            {
                "engine_id": engine_id,
                "cycle": cycle,
                "baseline_rul": expected_value,
                "predicted_rul": predicted_rul,
                "top_rul_reducing_features": (
                    ", ".join(
                        top_negative_features
                    )
                    if top_negative_features
                    else "None identified"
                ),
                "top_rul_increasing_features": (
                    ", ".join(
                        top_positive_features
                    )
                    if top_positive_features
                    else "None identified"
                ),
            }
        )

    explanation_dataframe = pd.DataFrame(
        explanation_records
    )

    summary_dataframe = pd.DataFrame(
        summary_records
    )

    return (
        explanation_dataframe,
        summary_dataframe,
    )


def plot_global_feature_importance(
    importance_dataframe,
):
    """
    Save a global SHAP feature-importance bar chart.
    """

    plot_data = (
        importance_dataframe
        .head(TOP_GLOBAL_FEATURES)
        .sort_values(
            "mean_absolute_shap",
            ascending=True,
        )
    )

    plt.figure(figsize=(11, 9))

    plt.barh(
        plot_data["feature"],
        plot_data["mean_absolute_shap"],
        color="#2874A6",
    )

    plt.title(
        "Global SHAP Feature Importance for RUL Prediction"
    )

    plt.xlabel(
        "Mean Absolute SHAP Contribution "
        "in Predicted RUL Cycles"
    )

    plt.ylabel("Model Feature")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY
        / "global_shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_shap_beeswarm(
    feature_dataframe,
    shap_values,
):
    """
    Save a SHAP beeswarm plot for fleet-level explanations.
    """

    explanation = shap.Explanation(
        values=shap_values,
        data=feature_dataframe.to_numpy(),
        feature_names=list(
            feature_dataframe.columns
        ),
    )

    shap.plots.beeswarm(
        explanation,
        max_display=TOP_GLOBAL_FEATURES,
        show=False,
    )

    plt.title(
        "Global SHAP Effect Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY
        / "global_shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_explainability_outputs(
    global_importance,
    local_explanations,
    local_summaries,
    metadata,
):
    """
    Save explainability reports and metadata.
    """

    global_importance.to_csv(
        OUTPUT_DIRECTORY
        / "global_feature_importance.csv",
        index=False,
    )

    local_explanations.to_csv(
        OUTPUT_DIRECTORY
        / "local_asset_explanations.csv",
        index=False,
    )

    local_summaries.to_csv(
        OUTPUT_DIRECTORY
        / "local_explanation_summary.csv",
        index=False,
    )

    with (
        OUTPUT_DIRECTORY
        / "explainability_metadata.json"
    ).open(
        mode="w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
        )


def run_model_explainability():
    """
    Run fleet-level and asset-level SHAP explainability.
    """

    prepare_output_directory()

    model = load_rul_model()
    feature_columns = load_model_features()
    test_dataframe = load_test_feature_data()

    validate_model_features(
        test_dataframe,
        feature_columns,
    )

    global_sample = create_global_sample(
        test_dataframe
    )

    final_cycle_dataframe = (
        get_final_cycle_data(
            test_dataframe
        )
    )

    explainer = create_tree_explainer(
        model
    )

    global_feature_dataframe = (
        global_sample[
            feature_columns
        ]
    )

    final_cycle_feature_dataframe = (
        final_cycle_dataframe[
            feature_columns
        ]
    )

    print("\nCalculating global SHAP explanations...")

    global_shap_values = (
        calculate_shap_values(
            explainer,
            global_feature_dataframe,
        )
    )

    print(
        "Calculating final-cycle asset explanations..."
    )

    final_cycle_shap_values = (
        calculate_shap_values(
            explainer,
            final_cycle_feature_dataframe,
        )
    )

    expected_value = extract_expected_value(
        explainer
    )

    global_additivity_difference = (
        validate_shap_additivity(
            model,
            global_feature_dataframe,
            global_shap_values,
            expected_value,
        )
    )

    local_additivity_difference = (
        validate_shap_additivity(
            model,
            final_cycle_feature_dataframe,
            final_cycle_shap_values,
            expected_value,
        )
    )

    global_importance = (
        create_global_importance(
            global_shap_values,
            feature_columns,
        )
    )

    (
        local_explanations,
        local_summaries,
    ) = create_local_explanations(
        final_cycle_dataframe,
        final_cycle_shap_values,
        feature_columns,
        expected_value,
        model,
    )

    metadata = {
        "explainer": "SHAP TreeExplainer",
        "model_type": type(model).__name__,
        "expected_value": expected_value,
        "global_sample_size": len(
            global_sample
        ),
        "locally_explained_assets": int(
            final_cycle_dataframe[
                "engine_id"
            ].nunique()
        ),
        "number_of_features": len(
            feature_columns
        ),
        "maximum_global_additivity_difference": (
            global_additivity_difference
        ),
        "maximum_local_additivity_difference": (
            local_additivity_difference
        ),
        "interpretation": {
            "positive_shap": (
                "Feature increases the predicted RUL."
            ),
            "negative_shap": (
                "Feature reduces the predicted RUL."
            ),
            "absolute_shap": (
                "Magnitude of feature influence on "
                "the model prediction."
            ),
        },
        "limitations": [
            (
                "SHAP explains the fitted model's behaviour "
                "and does not establish causal relationships."
            ),
            (
                "Correlated rolling features may share or "
                "redistribute attribution."
            ),
            (
                "Explanations are based on simulated NASA "
                "C-MAPSS data."
            ),
            (
                "The model is not validated for real-world "
                "safety-critical maintenance use."
            ),
        ],
    }

    save_explainability_outputs(
        global_importance,
        local_explanations,
        local_summaries,
        metadata,
    )

    plot_global_feature_importance(
        global_importance
    )

    plot_shap_beeswarm(
        global_feature_dataframe,
        global_shap_values,
    )

    print("\nRUL MODEL EXPLAINABILITY")
    print("=" * 75)

    print(
        f"Expected baseline RUL: "
        f"{expected_value:.3f} cycles"
    )

    print(
        f"Global observations explained: "
        f"{len(global_sample)}"
    )

    print(
        f"Final-cycle assets explained: "
        f"{len(final_cycle_dataframe)}"
    )

    print(
        f"Maximum additivity difference: "
        f"{max(
            global_additivity_difference,
            local_additivity_difference
        ):.8f}"
    )

    print("\nTOP GLOBAL RUL FEATURES")
    print("-" * 75)

    print(
        global_importance[
            [
                "importance_rank",
                "feature",
                "mean_absolute_shap",
                "mean_signed_shap",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print(
        f"\nOutputs saved to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    run_model_explainability()