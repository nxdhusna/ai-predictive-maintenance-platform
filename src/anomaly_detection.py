from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.data_loader import PROCESSED_DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FEATURE_PATH = (
    PROCESSED_DATA_DIR
    / "train_FD001_features.csv"
)

TEST_FEATURE_PATH = (
    PROCESSED_DATA_DIR
    / "test_FD001_features.csv"
)

RUL_MODEL_FEATURE_PATH = (
    PROJECT_ROOT
    / "models"
    / "rul_model_features.txt"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "anomaly_detection"
)

HEALTHY_RUL_THRESHOLD = 60
CONTAMINATION_RATE = 0.05
RANDOM_STATE = 42


def prepare_output_directories():
    """
    Create model and output directories.
    """

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_feature_datasets():
    """
    Load feature-engineered training and test datasets.
    """

    if not TRAIN_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "Training feature data was not found. "
            "Run 'python -m src.feature_engineering' first."
        )

    if not TEST_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "Test feature data was not found. "
            "Run 'python -m src.feature_engineering' first."
        )

    training_dataframe = pd.read_csv(
        TRAIN_FEATURE_PATH
    )

    test_dataframe = pd.read_csv(
        TEST_FEATURE_PATH
    )

    required_columns = {
        "engine_id",
        "cycle",
        "rul",
    }

    for dataframe_name, dataframe in [
        ("Training data", training_dataframe),
        ("Test data", test_dataframe),
    ]:
        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{dataframe_name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

    return training_dataframe, test_dataframe


def load_anomaly_feature_columns(dataframe):
    """
    Load model features and retain sensor-related features.

    Cycle and operational-setting columns are excluded because
    the anomaly detector should focus primarily on sensor
    behaviour rather than the age of the engine.
    """

    if not RUL_MODEL_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "The RUL model feature list was not found. "
            "Run 'python -m src.model_training' first."
        )

    with RUL_MODEL_FEATURE_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as feature_file:
        feature_columns = [
            line.strip()
            for line in feature_file
            if line.strip()
        ]

    prohibited_features = {
        "engine_id",
        "cycle",
        "rul",
        "rul_capped",
        "log_cycle",
        "observed_life_fraction",
        "operational_setting_1",
        "operational_setting_2",
        "operational_setting_3",
    }

    anomaly_features = [
        feature
        for feature in feature_columns
        if (
            feature not in prohibited_features
            and feature.startswith("sensor_")
        )
    ]

    missing_features = [
        feature
        for feature in anomaly_features
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Anomaly-detection features are missing: "
            f"{missing_features}"
        )

    if not anomaly_features:
        raise ValueError(
            "No sensor-related anomaly features were found."
        )

    return anomaly_features


def select_healthy_reference_data(
    training_dataframe,
):
    """
    Select observations representing relatively healthy operation.

    Rows with more than 60 cycles of actual remaining life are
    used as the reference distribution.
    """

    healthy_dataframe = (
        training_dataframe.loc[
            training_dataframe["rul"]
            > HEALTHY_RUL_THRESHOLD
        ]
        .copy()
        .reset_index(drop=True)
    )

    if healthy_dataframe.empty:
        raise ValueError(
            "No healthy reference observations were found."
        )

    if (
        healthy_dataframe["rul"]
        <= HEALTHY_RUL_THRESHOLD
    ).any():
        raise ValueError(
            "Unhealthy observations entered the "
            "healthy reference dataset."
        )

    return healthy_dataframe


def train_anomaly_detector(
    healthy_dataframe,
    feature_columns,
):
    """
    Train RobustScaler and Isolation Forest on healthy observations.
    """

    scaler = RobustScaler()

    healthy_scaled_features = (
        scaler.fit_transform(
            healthy_dataframe[
                feature_columns
            ]
        )
    )

    detector = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=CONTAMINATION_RATE,
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    detector.fit(
        healthy_scaled_features
    )

    healthy_decision_scores = (
        detector.decision_function(
            healthy_scaled_features
        )
    )

    score_reference = {
        "minimum": float(
            np.min(healthy_decision_scores)
        ),
        "maximum": float(
            np.max(healthy_decision_scores)
        ),
        "median": float(
            np.median(healthy_decision_scores)
        ),
        "percentile_1": float(
            np.percentile(
                healthy_decision_scores,
                1,
            )
        ),
        "percentile_5": float(
            np.percentile(
                healthy_decision_scores,
                5,
            )
        ),
        "percentile_95": float(
            np.percentile(
                healthy_decision_scores,
                95,
            )
        ),
    }

    return (
        scaler,
        detector,
        healthy_decision_scores,
        score_reference,
    )


def convert_to_anomaly_score(
    decision_scores,
    healthy_reference_scores,
):
    """
    Convert Isolation Forest decision values to a 0-100 score.

    Lower decision-function values indicate more unusual
    observations. The score is calculated using the percentile
    position relative to healthy reference scores.

    A higher score means a more unusual sensor pattern.
    """

    decision_scores = np.asarray(
        decision_scores,
        dtype=float,
    )

    healthy_reference_scores = np.sort(
        np.asarray(
            healthy_reference_scores,
            dtype=float,
        )
    )

    percentile_positions = np.searchsorted(
        healthy_reference_scores,
        decision_scores,
        side="right",
    ) / len(healthy_reference_scores)

    anomaly_scores = (
        1.0 - percentile_positions
    ) * 100

    return np.round(
        np.clip(
            anomaly_scores,
            0,
            100,
        ),
        1,
    )


def assign_anomaly_severity(
    anomaly_score,
):
    """
    Assign an interpretable anomaly-severity label.
    """

    if anomaly_score >= 95:
        return "Critical"

    if anomaly_score >= 80:
        return "High"

    if anomaly_score >= 60:
        return "Moderate"

    return "Normal"


def assign_anomaly_status(
    isolation_forest_prediction,
):
    """
    Convert Isolation Forest output into a readable label.

    Isolation Forest returns:
    -1 for an anomaly
     1 for an inlier
    """

    if isolation_forest_prediction == -1:
        return "Anomaly Detected"

    return "Normal Behaviour"


def explain_anomaly_result(
    anomaly_status,
    anomaly_score,
    anomaly_severity,
):
    """
    Create a concise explanation for the dashboard.
    """

    if anomaly_status == "Anomaly Detected":
        return (
            f"The sensor pattern is unusually different from "
            f"the healthy reference distribution. The anomaly "
            f"score is {anomaly_score:.1f}/100 with "
            f"{anomaly_severity.lower()} severity."
        )

    return (
        f"The sensor pattern remains within the expected healthy "
        f"reference distribution. The anomaly score is "
        f"{anomaly_score:.1f}/100."
    )


def score_anomalies(
    dataframe,
    feature_columns,
    scaler,
    detector,
    healthy_reference_scores,
):
    """
    Generate anomaly results for every observation.
    """

    result = dataframe.copy()

    scaled_features = scaler.transform(
        result[feature_columns]
    )

    decision_scores = (
        detector.decision_function(
            scaled_features
        )
    )

    isolation_predictions = (
        detector.predict(
            scaled_features
        )
    )

    result[
        "anomaly_decision_score"
    ] = decision_scores

    result[
        "anomaly_score"
    ] = convert_to_anomaly_score(
        decision_scores,
        healthy_reference_scores,
    )

    result[
        "is_anomaly"
    ] = (
        isolation_predictions == -1
    ).astype(int)

    result[
        "anomaly_status"
    ] = [
        assign_anomaly_status(prediction)
        for prediction in isolation_predictions
    ]

    result[
        "anomaly_severity"
    ] = (
        result["anomaly_score"]
        .apply(assign_anomaly_severity)
    )

    result[
        "anomaly_explanation"
    ] = result.apply(
        lambda row: explain_anomaly_result(
            row["anomaly_status"],
            row["anomaly_score"],
            row["anomaly_severity"],
        ),
        axis=1,
    )

    return result


def add_rul_bands(dataframe):
    """
    Add lifecycle bands for anomaly-rate analysis.
    """

    result = dataframe.copy()

    result["rul_band"] = pd.cut(
        result["rul"],
        bins=[
            -np.inf,
            10,
            30,
            60,
            125,
            np.inf,
        ],
        labels=[
            "0-10 cycles",
            "11-30 cycles",
            "31-60 cycles",
            "61-125 cycles",
            "Above 125 cycles",
        ],
    )

    return result


def create_anomaly_rate_summary(
    dataframe,
):
    """
    Calculate anomaly rate and mean score by RUL band.
    """

    result = add_rul_bands(
        dataframe
    )

    summary = (
        result.groupby(
            "rul_band",
            observed=False,
        )
        .agg(
            observation_count=(
                "is_anomaly",
                "size",
            ),
            anomaly_count=(
                "is_anomaly",
                "sum",
            ),
            anomaly_rate=(
                "is_anomaly",
                "mean",
            ),
            average_anomaly_score=(
                "anomaly_score",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["anomaly_rate"] = (
        summary["anomaly_rate"] * 100
    )

    return summary


def get_final_cycle_results(
    scored_test_dataframe,
):
    """
    Select the final observed cycle of each test engine.
    """

    final_cycle_results = (
        scored_test_dataframe
        .sort_values(
            ["engine_id", "cycle"]
        )
        .groupby("engine_id")
        .tail(1)
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    final_cycle_results[
        "anomaly_priority_rank"
    ] = range(
        1,
        len(final_cycle_results) + 1,
    )

    return final_cycle_results


def validate_anomaly_results(
    dataframe,
):
    """
    Validate anomaly outputs before they are saved.
    """

    required_columns = {
        "anomaly_decision_score",
        "anomaly_score",
        "is_anomaly",
        "anomaly_status",
        "anomaly_severity",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Anomaly outputs are missing: "
            f"{sorted(missing_columns)}"
        )

    if dataframe[list(required_columns)].isna().any().any():
        raise ValueError(
            "Anomaly outputs contain missing values."
        )

    if not dataframe["anomaly_score"].between(
        0,
        100,
    ).all():
        raise ValueError(
            "Anomaly scores must remain between 0 and 100."
        )

    valid_predictions = {0, 1}

    if not set(
        dataframe["is_anomaly"].unique()
    ).issubset(valid_predictions):
        raise ValueError(
            "Anomaly flags must use only 0 and 1."
        )


def create_anomaly_summary(
    scored_training_dataframe,
    scored_test_dataframe,
    final_cycle_results,
    feature_columns,
    healthy_reference_rows,
):
    """
    Create summary metadata for the anomaly detector.
    """

    return {
        "algorithm": "Isolation Forest",
        "healthy_rul_threshold": (
            HEALTHY_RUL_THRESHOLD
        ),
        "configured_contamination_rate": (
            CONTAMINATION_RATE
        ),
        "healthy_reference_observations": int(
            healthy_reference_rows
        ),
        "number_of_features": int(
            len(feature_columns)
        ),
        "training_observations_scored": int(
            len(scored_training_dataframe)
        ),
        "test_observations_scored": int(
            len(scored_test_dataframe)
        ),
        "training_anomaly_rate_percent": round(
            float(
                scored_training_dataframe[
                    "is_anomaly"
                ].mean()
                * 100
            ),
            2,
        ),
        "test_anomaly_rate_percent": round(
            float(
                scored_test_dataframe[
                    "is_anomaly"
                ].mean()
                * 100
            ),
            2,
        ),
        "final_cycle_anomalous_assets": int(
            final_cycle_results[
                "is_anomaly"
            ].sum()
        ),
        "final_cycle_average_anomaly_score": round(
            float(
                final_cycle_results[
                    "anomaly_score"
                ].mean()
            ),
            2,
        ),
        "methodology_note": (
            "The anomaly score measures unusual sensor "
            "behaviour relative to observations with more than "
            "60 remaining cycles. It is not a probability of "
            "failure."
        ),
    }


def plot_anomaly_score_distribution(
    scored_test_dataframe,
):
    """
    Plot anomaly-score distributions by detector status.
    """

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=scored_test_dataframe,
        x="anomaly_score",
        hue="anomaly_status",
        bins=25,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.35,
    )

    plt.title(
        "Test Sensor Anomaly Score Distribution"
    )

    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY
        / "anomaly_score_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_anomaly_rate_by_rul_band(
    anomaly_rate_summary,
):
    """
    Plot anomaly rate across degradation stages.
    """

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=anomaly_rate_summary,
        x="rul_band",
        y="anomaly_rate",
    )

    plt.title(
        "Anomaly Rate by Remaining Useful Life Band"
    )

    plt.xlabel("Remaining Useful Life Band")
    plt.ylabel("Anomaly Rate (%)")

    plt.xticks(
        rotation=15,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY
        / "anomaly_rate_by_rul_band.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_anomaly_score_vs_rul(
    scored_test_dataframe,
):
    """
    Plot the relationship between RUL and anomaly score.
    """

    plot_data = scored_test_dataframe.sample(
        n=min(
            5000,
            len(scored_test_dataframe),
        ),
        random_state=RANDOM_STATE,
    )

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=plot_data,
        x="rul",
        y="anomaly_score",
        hue="anomaly_status",
        alpha=0.45,
        s=24,
    )

    plt.title(
        "Sensor Anomaly Score vs Remaining Useful Life"
    )

    plt.xlabel("Actual Remaining Useful Life")
    plt.ylabel("Anomaly Score")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY
        / "anomaly_score_vs_rul.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_top_anomalous_engines(
    final_cycle_results,
    number_of_engines=20,
):
    """
    Visualise the most anomalous final-cycle engine readings.
    """

    plot_data = (
        final_cycle_results
        .head(number_of_engines)
        .sort_values(
            "anomaly_score",
            ascending=True,
        )
        .copy()
    )

    plot_data["engine_label"] = (
        "Engine "
        + plot_data["engine_id"]
        .astype(int)
        .astype(str)
    )

    plt.figure(figsize=(11, 8))

    sns.barplot(
        data=plot_data,
        x="anomaly_score",
        y="engine_label",
        hue="anomaly_severity",
        dodge=False,
    )

    plt.title(
        "Most Anomalous Test Engines at Final Observation"
    )

    plt.xlabel("Anomaly Score")
    plt.ylabel("Asset")
    plt.xlim(0, 100)
    plt.legend(title="Severity")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY
        / "top_anomalous_engines.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_anomaly_detector(
    scaler,
    detector,
    healthy_reference_scores,
    score_reference,
    feature_columns,
):
    """
    Save the fitted anomaly-detection bundle and metadata.
    """

    bundle = {
        "scaler": scaler,
        "detector": detector,
        "healthy_reference_scores": (
            healthy_reference_scores
        ),
        "feature_columns": feature_columns,
        "healthy_rul_threshold": (
            HEALTHY_RUL_THRESHOLD
        ),
    }

    model_path = (
        MODEL_DIRECTORY
        / "anomaly_detector.joblib"
    )

    joblib.dump(
        bundle,
        model_path,
    )

    feature_path = (
        MODEL_DIRECTORY
        / "anomaly_detector_features.txt"
    )

    with feature_path.open(
        mode="w",
        encoding="utf-8",
    ) as feature_file:
        for feature in feature_columns:
            feature_file.write(
                f"{feature}\n"
            )

    metadata = {
        "algorithm": "Isolation Forest",
        "healthy_reference_definition": (
            "Training observations with actual RUL "
            "greater than 60 cycles."
        ),
        "healthy_rul_threshold": (
            HEALTHY_RUL_THRESHOLD
        ),
        "contamination_rate": (
            CONTAMINATION_RATE
        ),
        "number_of_estimators": 300,
        "number_of_features": len(
            feature_columns
        ),
        "score_reference": score_reference,
        "anomaly_score_interpretation": {
            "0_to_59": "Normal",
            "60_to_79": "Moderate",
            "80_to_94": "High",
            "95_to_100": "Critical",
        },
        "limitations": [
            (
                "The detector is trained using simulated "
                "NASA C-MAPSS data."
            ),
            (
                "Anomalies indicate unusual multivariate "
                "sensor patterns, not confirmed faults."
            ),
            (
                "The anomaly score is not a calibrated "
                "failure probability."
            ),
            (
                "Operational thresholds require domain "
                "expert validation before real deployment."
            ),
        ],
    }

    metadata_path = (
        MODEL_DIRECTORY
        / "anomaly_detector_metadata.json"
    )

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
        )

    return model_path


def save_outputs(
    scored_test_dataframe,
    final_cycle_results,
    anomaly_rate_summary,
    anomaly_summary,
):
    """
    Save detailed results and summary files.
    """

    scored_test_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "anomaly_detection_results.csv",
        index=False,
    )

    final_cycle_results.to_csv(
        OUTPUT_DIRECTORY
        / "final_cycle_anomaly_results.csv",
        index=False,
    )

    anomaly_rate_summary.to_csv(
        OUTPUT_DIRECTORY
        / "anomaly_rate_by_rul_band.csv",
        index=False,
    )

    with (
        OUTPUT_DIRECTORY
        / "anomaly_summary.json"
    ).open(
        mode="w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            anomaly_summary,
            summary_file,
            indent=4,
        )


def run_anomaly_detection():
    """
    Run the complete sensor anomaly-detection pipeline.
    """

    prepare_output_directories()

    (
        training_dataframe,
        test_dataframe,
    ) = load_feature_datasets()

    feature_columns = (
        load_anomaly_feature_columns(
            training_dataframe
        )
    )

    healthy_dataframe = (
        select_healthy_reference_data(
            training_dataframe
        )
    )

    (
        scaler,
        detector,
        healthy_reference_scores,
        score_reference,
    ) = train_anomaly_detector(
        healthy_dataframe,
        feature_columns,
    )

    scored_training_dataframe = (
        score_anomalies(
            training_dataframe,
            feature_columns,
            scaler,
            detector,
            healthy_reference_scores,
        )
    )

    scored_test_dataframe = (
        score_anomalies(
            test_dataframe,
            feature_columns,
            scaler,
            detector,
            healthy_reference_scores,
        )
    )

    validate_anomaly_results(
        scored_training_dataframe
    )

    validate_anomaly_results(
        scored_test_dataframe
    )

    anomaly_rate_summary = (
        create_anomaly_rate_summary(
            scored_test_dataframe
        )
    )

    final_cycle_results = (
        get_final_cycle_results(
            scored_test_dataframe
        )
    )

    anomaly_summary = (
        create_anomaly_summary(
            scored_training_dataframe,
            scored_test_dataframe,
            final_cycle_results,
            feature_columns,
            len(healthy_dataframe),
        )
    )

    save_outputs(
        scored_test_dataframe,
        final_cycle_results,
        anomaly_rate_summary,
        anomaly_summary,
    )

    plot_anomaly_score_distribution(
        scored_test_dataframe
    )

    plot_anomaly_rate_by_rul_band(
        anomaly_rate_summary
    )

    plot_anomaly_score_vs_rul(
        scored_test_dataframe
    )

    plot_top_anomalous_engines(
        final_cycle_results
    )

    model_path = save_anomaly_detector(
        scaler,
        detector,
        healthy_reference_scores,
        score_reference,
        feature_columns,
    )

    print("\nSENSOR ANOMALY DETECTION")
    print("=" * 75)

    print(
        f"Healthy reference observations: "
        f"{len(healthy_dataframe)}"
    )

    print(
        f"Anomaly-detection features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Training anomaly rate: "
        f"{anomaly_summary['training_anomaly_rate_percent']}%"
    )

    print(
        f"Test anomaly rate: "
        f"{anomaly_summary['test_anomaly_rate_percent']}%"
    )

    print(
        f"Final-cycle anomalous assets: "
        f"{anomaly_summary['final_cycle_anomalous_assets']}"
    )

    print(
        f"Average final-cycle anomaly score: "
        f"{anomaly_summary['final_cycle_average_anomaly_score']}/100"
    )

    print("\nANOMALY RATE BY RUL BAND")
    print("-" * 75)

    print(
        anomaly_rate_summary.to_string(
            index=False
        )
    )

    print("\nTOP ANOMALOUS FINAL-CYCLE ASSETS")
    print("-" * 75)

    display_columns = [
        "anomaly_priority_rank",
        "engine_id",
        "cycle",
        "rul",
        "anomaly_score",
        "anomaly_severity",
        "anomaly_status",
    ]

    print(
        final_cycle_results[
            display_columns
        ]
        .head(10)
        .to_string(index=False)
    )

    print(
        f"\nSaved detector: {model_path}"
    )

    print(
        f"Outputs saved to: "
        f"{OUTPUT_DIRECTORY}"
    )


if __name__ == "__main__":
    run_anomaly_detection()