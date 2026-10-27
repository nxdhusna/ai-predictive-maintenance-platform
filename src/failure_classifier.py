from pathlib import Path
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import PROCESSED_DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FEATURE_PATH = (
    PROCESSED_DATA_DIR
    / "train_FD001_features.csv"
)

MODEL_FEATURE_PATH = (
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
    / "failure_classification"
)

FAILURE_HORIZON = 30
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.50

GROUP_COLUMN = "engine_id"
TARGET_COLUMN = "failure_within_30_cycles"


def prepare_output_directories():
    """
    Create model and evaluation output directories.
    """

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_training_data():
    """
    Load the feature-engineered FD001 training data.
    """

    if not TRAIN_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "Training feature data was not found. "
            "Run 'python -m src.feature_engineering' first."
        )

    dataframe = pd.read_csv(
        TRAIN_FEATURE_PATH
    )

    required_columns = {
        "engine_id",
        "cycle",
        "rul",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def load_feature_columns(dataframe):
    """
    Load the feature list already used by the RUL model.
    """

    if not MODEL_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "The saved model-feature list was not found. "
            "Run 'python -m src.model_training' first."
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

    prohibited_features = {
        "engine_id",
        "rul",
        "rul_capped",
        "failure_within_30_cycles",
        "observed_life_fraction",
    }

    feature_columns = [
        feature
        for feature in feature_columns
        if feature not in prohibited_features
    ]

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Training data is missing model features: "
            f"{missing_features}"
        )

    if not feature_columns:
        raise ValueError(
            "No classifier features are available."
        )

    return feature_columns


def add_failure_target(
    dataframe,
    failure_horizon=FAILURE_HORIZON,
):
    """
    Create the binary classification target.

    Positive class:
    The engine has 30 or fewer actual cycles remaining.

    Negative class:
    The engine has more than 30 cycles remaining.
    """

    result = dataframe.copy()

    result[TARGET_COLUMN] = (
        result["rul"] <= failure_horizon
    ).astype(int)

    return result


def split_engine_groups(
    dataframe,
):
    """
    Create separate training, calibration, and validation groups.

    First split:
    80% development engines and 20% final validation engines.

    Second split:
    75% of development engines become model-training engines.
    25% become calibration engines.

    This produces approximately:
    60 training engines
    20 calibration engines
    20 validation engines.
    """

    development_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    development_indices, validation_indices = next(
        development_splitter.split(
            dataframe,
            groups=dataframe[GROUP_COLUMN],
        )
    )

    development_dataframe = (
        dataframe
        .iloc[development_indices]
        .copy()
        .reset_index(drop=True)
    )

    validation_dataframe = (
        dataframe
        .iloc[validation_indices]
        .copy()
        .reset_index(drop=True)
    )

    calibration_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    training_indices, calibration_indices = next(
        calibration_splitter.split(
            development_dataframe,
            groups=development_dataframe[
                GROUP_COLUMN
            ],
        )
    )

    training_dataframe = (
        development_dataframe
        .iloc[training_indices]
        .copy()
        .reset_index(drop=True)
    )

    calibration_dataframe = (
        development_dataframe
        .iloc[calibration_indices]
        .copy()
        .reset_index(drop=True)
    )

    validate_group_separation(
        training_dataframe,
        calibration_dataframe,
        validation_dataframe,
    )

    return (
        training_dataframe,
        calibration_dataframe,
        validation_dataframe,
    )


def validate_group_separation(
    training_dataframe,
    calibration_dataframe,
    validation_dataframe,
):
    """
    Confirm that no engine occurs in more than one split.
    """

    training_engines = set(
        training_dataframe[
            GROUP_COLUMN
        ].unique()
    )

    calibration_engines = set(
        calibration_dataframe[
            GROUP_COLUMN
        ].unique()
    )

    validation_engines = set(
        validation_dataframe[
            GROUP_COLUMN
        ].unique()
    )

    train_calibration_overlap = (
        training_engines
        & calibration_engines
    )

    train_validation_overlap = (
        training_engines
        & validation_engines
    )

    calibration_validation_overlap = (
        calibration_engines
        & validation_engines
    )

    if train_calibration_overlap:
        raise ValueError(
            "Training and calibration engines overlap."
        )

    if train_validation_overlap:
        raise ValueError(
            "Training and validation engines overlap."
        )

    if calibration_validation_overlap:
        raise ValueError(
            "Calibration and validation engines overlap."
        )


def create_classifiers():
    """
    Create candidate binary classifiers.
    """

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=250,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            )
        ),
    }

    return models


def calculate_classification_metrics(
    actual_values,
    predicted_labels,
    predicted_probabilities,
):
    """
    Calculate discrimination, classification, and probability metrics.
    """

    return {
        "accuracy": float(
            accuracy_score(
                actual_values,
                predicted_labels,
            )
        ),
        "precision": float(
            precision_score(
                actual_values,
                predicted_labels,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual_values,
                predicted_labels,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                actual_values,
                predicted_labels,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                actual_values,
                predicted_probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                actual_values,
                predicted_probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                actual_values,
                predicted_probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                actual_values,
                predicted_probabilities,
                labels=[0, 1],
            )
        ),
    }


def train_candidate_classifiers(
    training_dataframe,
    calibration_dataframe,
    feature_columns,
):
    """
    Train candidate classifiers and evaluate their raw probabilities
    on the separate calibration-engine split.
    """

    x_train = training_dataframe[
        feature_columns
    ]

    y_train = training_dataframe[
        TARGET_COLUMN
    ]

    x_calibration = calibration_dataframe[
        feature_columns
    ]

    y_calibration = calibration_dataframe[
        TARGET_COLUMN
    ]

    models = create_classifiers()

    evaluation_records = []
    trained_models = {}

    for model_name, model in models.items():
        print(
            f"\nTraining classifier: "
            f"{model_name}"
        )

        start_time = time.perf_counter()

        model.fit(
            x_train,
            y_train,
        )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        probabilities = model.predict_proba(
            x_calibration
        )[:, 1]

        labels = (
            probabilities
            >= DECISION_THRESHOLD
        ).astype(int)

        metrics = calculate_classification_metrics(
            y_calibration,
            labels,
            probabilities,
        )

        evaluation_records.append(
            {
                "model": model_name,
                **metrics,
                "training_seconds": round(
                    training_seconds,
                    3,
                ),
            }
        )

        trained_models[model_name] = model

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.3f}"
        )

        print(
            f"Recall: "
            f"{metrics['recall']:.3f}"
        )

        print(
            f"F1: "
            f"{metrics['f1']:.3f}"
        )

        print(
            f"Brier score: "
            f"{metrics['brier_score']:.4f}"
        )

    comparison_dataframe = pd.DataFrame(
        evaluation_records
    )

    comparison_dataframe = (
        comparison_dataframe
        .sort_values(
            by=[
                "roc_auc",
                "average_precision",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return (
        comparison_dataframe,
        trained_models,
    )


def fit_sigmoid_calibrator(
    base_model,
    calibration_dataframe,
    feature_columns,
):
    """
    Fit a sigmoid probability calibrator.

    The base classifier produces an uncalibrated probability.
    A logistic-regression calibration model learns how to map
    that value onto a better event-probability estimate.

    Calibration engines are separate from model-training engines.
    """

    raw_probabilities = base_model.predict_proba(
        calibration_dataframe[
            feature_columns
        ]
    )[:, 1]

    calibration_inputs = raw_probabilities.reshape(
        -1,
        1,
    )

    calibration_targets = calibration_dataframe[
        TARGET_COLUMN
    ]

    calibrator = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(
        calibration_inputs,
        calibration_targets,
    )

    return calibrator


def predict_calibrated_probability(
    base_model,
    calibrator,
    feature_dataframe,
    feature_columns,
):
    """
    Generate calibrated failure probabilities.
    """

    raw_probabilities = base_model.predict_proba(
        feature_dataframe[
            feature_columns
        ]
    )[:, 1]

    calibrated_probabilities = (
        calibrator.predict_proba(
            raw_probabilities.reshape(
                -1,
                1,
            )
        )[:, 1]
    )

    calibrated_probabilities = np.clip(
        calibrated_probabilities,
        0.0,
        1.0,
    )

    return (
        raw_probabilities,
        calibrated_probabilities,
    )


def create_validation_predictions(
    validation_dataframe,
    base_model,
    calibrator,
    feature_columns,
):
    """
    Generate final held-out validation probabilities and labels.
    """

    (
        raw_probabilities,
        calibrated_probabilities,
    ) = predict_calibrated_probability(
        base_model,
        calibrator,
        validation_dataframe,
        feature_columns,
    )

    predicted_labels = (
        calibrated_probabilities
        >= DECISION_THRESHOLD
    ).astype(int)

    result = validation_dataframe[
        [
            "engine_id",
            "cycle",
            "rul",
            "rul_capped",
            TARGET_COLUMN,
        ]
    ].copy()

    result["raw_failure_probability"] = (
        raw_probabilities
    )

    result[
        "calibrated_failure_probability"
    ] = calibrated_probabilities

    result["predicted_failure_class"] = (
        predicted_labels
    )

    result["prediction_correct"] = (
        result["predicted_failure_class"]
        == result[TARGET_COLUMN]
    )

    return result


def calculate_probability_bins(
    actual_values,
    predicted_probabilities,
    number_of_bins=10,
):
    """
    Calculate a reliability table for the calibration curve.
    """

    probability_dataframe = pd.DataFrame(
        {
            "actual": actual_values,
            "probability": predicted_probabilities,
        }
    )

    probability_dataframe["probability_bin"] = pd.cut(
        probability_dataframe["probability"],
        bins=np.linspace(
            0,
            1,
            number_of_bins + 1,
        ),
        include_lowest=True,
        duplicates="drop",
    )

    calibration_table = (
        probability_dataframe
        .groupby(
            "probability_bin",
            observed=True,
        )
        .agg(
            mean_predicted_probability=(
                "probability",
                "mean",
            ),
            observed_failure_rate=(
                "actual",
                "mean",
            ),
            observation_count=(
                "actual",
                "size",
            ),
        )
        .reset_index()
    )

    return calibration_table


def plot_confusion_matrix(
    actual_values,
    predicted_labels,
):
    """
    Plot the final held-out confusion matrix.
    """

    matrix = confusion_matrix(
        actual_values,
        predicted_labels,
        labels=[0, 1],
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "More than 30 cycles",
            "Within 30 cycles",
        ],
        yticklabels=[
            "More than 30 cycles",
            "Within 30 cycles",
        ],
    )

    plt.title(
        "Failure-Within-30-Cycles Confusion Matrix"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY
        / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_probability_calibration(
    calibration_table,
):
    """
    Plot predicted probability against observed failure frequency.
    """

    plot_data = calibration_table.dropna(
        subset=[
            "mean_predicted_probability",
            "observed_failure_rate",
        ]
    )

    plt.figure(figsize=(8, 7))

    plt.plot(
        plot_data[
            "mean_predicted_probability"
        ],
        plot_data[
            "observed_failure_rate"
        ],
        marker="o",
        linewidth=2,
        label="Calibrated classifier",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Perfect calibration",
    )

    plt.title(
        "Failure Probability Calibration Curve"
    )

    plt.xlabel(
        "Mean Predicted Probability"
    )

    plt.ylabel(
        "Observed Failure-Within-30-Cycles Rate"
    )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY
        / "probability_calibration_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_roc_curve(
    actual_values,
    predicted_probabilities,
):
    """
    Plot the Receiver Operating Characteristic curve.
    """

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            actual_values,
            predicted_probabilities,
        )
    )

    auc_value = roc_auc_score(
        actual_values,
        predicted_probabilities,
    )

    plt.figure(figsize=(8, 7))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"ROC-AUC = {auc_value:.3f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Random classifier",
    )

    plt.title(
        "Failure Classifier ROC Curve"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY
        / "roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_precision_recall_curve(
    actual_values,
    predicted_probabilities,
):
    """
    Plot the precision-recall curve.
    """

    precision, recall, _ = precision_recall_curve(
        actual_values,
        predicted_probabilities,
    )

    average_precision = average_precision_score(
        actual_values,
        predicted_probabilities,
    )

    plt.figure(figsize=(8, 7))

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=(
            "Average Precision = "
            f"{average_precision:.3f}"
        ),
    )

    plt.title(
        "Failure Classifier Precision-Recall Curve"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY
        / "precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_probability_distribution(
    validation_predictions,
):
    """
    Compare probability distributions for positive and negative rows.
    """

    plot_data = validation_predictions.copy()

    plot_data["actual_status"] = (
        plot_data[TARGET_COLUMN]
        .map(
            {
                0: "More than 30 cycles",
                1: "Failure within 30 cycles",
            }
        )
    )

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=plot_data,
        x="calibrated_failure_probability",
        hue="actual_status",
        bins=20,
        stat="density",
        common_norm=False,
        element="step",
        alpha=0.35,
    )

    plt.axvline(
        DECISION_THRESHOLD,
        color="black",
        linestyle="--",
        label="Decision threshold",
    )

    plt.title(
        "Calibrated Failure Probability Distribution"
    )

    plt.xlabel(
        "Probability of Failure Within 30 Cycles"
    )

    plt.ylabel("Density")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY
        / "probability_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_classifier_bundle(
    base_model,
    calibrator,
    feature_columns,
    model_name,
    metrics,
    training_dataframe,
    calibration_dataframe,
    validation_dataframe,
):
    """
    Save the classifier, calibrator, feature list, and metadata.
    """

    bundle = {
        "base_model": base_model,
        "probability_calibrator": calibrator,
        "failure_horizon": FAILURE_HORIZON,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_columns": feature_columns,
    }

    model_path = (
        MODEL_DIRECTORY
        / "failure_classifier.joblib"
    )

    joblib.dump(
        bundle,
        model_path,
    )

    feature_path = (
        MODEL_DIRECTORY
        / "failure_classifier_features.txt"
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
        "model_name": model_name,
        "event_definition": (
            "Actual Remaining Useful Life is "
            "30 cycles or fewer."
        ),
        "failure_horizon_cycles": FAILURE_HORIZON,
        "decision_threshold": DECISION_THRESHOLD,
        "calibration_method": (
            "Sigmoid logistic calibration on "
            "separate engine groups."
        ),
        "number_of_features": len(
            feature_columns
        ),
        "training_engines": int(
            training_dataframe[
                GROUP_COLUMN
            ].nunique()
        ),
        "calibration_engines": int(
            calibration_dataframe[
                GROUP_COLUMN
            ].nunique()
        ),
        "validation_engines": int(
            validation_dataframe[
                GROUP_COLUMN
            ].nunique()
        ),
        "validation_metrics": metrics,
        "limitations": [
            (
                "Trained using simulated NASA "
                "C-MAPSS FD001 data."
            ),
            (
                "Probability calibration is validated "
                "on held-out simulated engine cycles."
            ),
            (
                "Not validated for real aviation or "
                "safety-critical operational use."
            ),
        ],
    }

    metadata_path = (
        MODEL_DIRECTORY
        / "failure_classifier_metadata.json"
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


def run_failure_classifier_training():
    """
    Run the full failure-classification and calibration pipeline.
    """

    prepare_output_directories()

    dataframe = load_training_data()

    dataframe = add_failure_target(
        dataframe
    )

    feature_columns = load_feature_columns(
        dataframe
    )

    (
        training_dataframe,
        calibration_dataframe,
        validation_dataframe,
    ) = split_engine_groups(
        dataframe
    )

    print("\nFAILURE-WITHIN-30-CYCLES CLASSIFIER")
    print("=" * 75)

    print(
        "Training engines: "
        f"{training_dataframe[GROUP_COLUMN].nunique()}"
    )

    print(
        "Calibration engines: "
        f"{calibration_dataframe[GROUP_COLUMN].nunique()}"
    )

    print(
        "Validation engines: "
        f"{validation_dataframe[GROUP_COLUMN].nunique()}"
    )

    print(
        "Positive training observations: "
        f"{training_dataframe[TARGET_COLUMN].sum()}"
    )

    print(
        "Negative training observations: "
        f"{len(training_dataframe) - training_dataframe[TARGET_COLUMN].sum()}"
    )

    (
        comparison_dataframe,
        trained_models,
    ) = train_candidate_classifiers(
        training_dataframe,
        calibration_dataframe,
        feature_columns,
    )

    best_model_name = (
        comparison_dataframe
        .iloc[0]["model"]
    )

    best_base_model = trained_models[
        best_model_name
    ]

    calibrator = fit_sigmoid_calibrator(
        best_base_model,
        calibration_dataframe,
        feature_columns,
    )

    validation_predictions = (
        create_validation_predictions(
            validation_dataframe,
            best_base_model,
            calibrator,
            feature_columns,
        )
    )

    actual_values = validation_predictions[
        TARGET_COLUMN
    ]

    predicted_labels = validation_predictions[
        "predicted_failure_class"
    ]

    predicted_probabilities = validation_predictions[
        "calibrated_failure_probability"
    ]

    final_metrics = (
        calculate_classification_metrics(
            actual_values,
            predicted_labels,
            predicted_probabilities,
        )
    )

    probability_bins = (
        calculate_probability_bins(
            actual_values,
            predicted_probabilities,
        )
    )

    classification_report_dictionary = (
        classification_report(
            actual_values,
            predicted_labels,
            output_dict=True,
            zero_division=0,
        )
    )

    comparison_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "classifier_comparison.csv",
        index=False,
    )

    validation_predictions.to_csv(
        OUTPUT_DIRECTORY
        / "validation_predictions.csv",
        index=False,
    )

    probability_bins.to_csv(
        OUTPUT_DIRECTORY
        / "probability_calibration_table.csv",
        index=False,
    )

    metrics_output = {
        "selected_model": best_model_name,
        "failure_horizon_cycles": FAILURE_HORIZON,
        "decision_threshold": DECISION_THRESHOLD,
        "validation_metrics": final_metrics,
        "classification_report": (
            classification_report_dictionary
        ),
    }

    with (
        OUTPUT_DIRECTORY
        / "classification_metrics.json"
    ).open(
        mode="w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics_output,
            metrics_file,
            indent=4,
        )

    plot_confusion_matrix(
        actual_values,
        predicted_labels,
    )

    plot_probability_calibration(
        probability_bins
    )

    plot_roc_curve(
        actual_values,
        predicted_probabilities,
    )

    plot_precision_recall_curve(
        actual_values,
        predicted_probabilities,
    )

    plot_probability_distribution(
        validation_predictions
    )

    model_path = save_classifier_bundle(
        best_base_model,
        calibrator,
        feature_columns,
        best_model_name,
        final_metrics,
        training_dataframe,
        calibration_dataframe,
        validation_dataframe,
    )

    print("\nCLASSIFIER COMPARISON")
    print("-" * 75)

    print(
        comparison_dataframe.to_string(
            index=False
        )
    )

    print("\nSELECTED MODEL")
    print("-" * 75)

    print(
        f"Model: {best_model_name}"
    )

    print(
        f"Accuracy: "
        f"{final_metrics['accuracy']:.3f}"
    )

    print(
        f"Precision: "
        f"{final_metrics['precision']:.3f}"
    )

    print(
        f"Recall: "
        f"{final_metrics['recall']:.3f}"
    )

    print(
        f"F1 score: "
        f"{final_metrics['f1']:.3f}"
    )

    print(
        f"ROC-AUC: "
        f"{final_metrics['roc_auc']:.3f}"
    )

    print(
        f"Average Precision: "
        f"{final_metrics['average_precision']:.3f}"
    )

    print(
        f"Brier score: "
        f"{final_metrics['brier_score']:.4f}"
    )

    print(
        f"Log loss: "
        f"{final_metrics['log_loss']:.4f}"
    )

    print(
        f"\nSaved classifier: "
        f"{model_path}"
    )


if __name__ == "__main__":
    run_failure_classifier_training()