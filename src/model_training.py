from pathlib import Path
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import PROCESSED_DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIRECTORY = PROJECT_ROOT / "models"

EVALUATION_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "model_evaluation"
)

TRAIN_FEATURE_PATH = (
    PROCESSED_DATA_DIR / "train_FD001_features.csv"
)

TEST_FEATURE_PATH = (
    PROCESSED_DATA_DIR / "test_FD001_features.csv"
)

FEATURE_LIST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "eda"
    / "model_feature_columns.txt"
)

TARGET_COLUMN = "rul_capped"
GROUP_COLUMN = "engine_id"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


def prepare_output_directories():
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_feature_dataframe():
    if not TRAIN_FEATURE_PATH.exists():
        raise FileNotFoundError(
            "Training feature file was not found. "
            "Run 'python -m src.feature_engineering' first."
        )

    dataframe = pd.read_csv(
        TRAIN_FEATURE_PATH
    )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    return dataframe


def load_model_feature_columns(dataframe):
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(
            "Model feature list was not found. "
            "Run the feature-engineering stage first."
        )

    with FEATURE_LIST_PATH.open(
        mode="r",
        encoding="utf-8",
    ) as feature_file:
        feature_columns = [
            line.strip()
            for line in feature_file
            if line.strip()
        ]

    excluded_features = {
        "engine_id",
        "rul",
        "rul_capped",
        "observed_life_fraction",
    }

    feature_columns = [
        feature
        for feature in feature_columns
        if feature not in excluded_features
    ]

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "The following model features are missing: "
            f"{missing_features}"
        )

    if not feature_columns:
        raise ValueError(
            "No valid model features were found."
        )

    return feature_columns


def create_grouped_train_validation_split(
    dataframe,
    feature_columns,
):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    features = dataframe[feature_columns]
    target = dataframe[TARGET_COLUMN]
    groups = dataframe[GROUP_COLUMN]

    train_indices, validation_indices = next(
        splitter.split(
            features,
            target,
            groups=groups,
        )
    )

    training_dataframe = (
        dataframe.iloc[train_indices]
        .copy()
        .reset_index(drop=True)
    )

    validation_dataframe = (
        dataframe.iloc[validation_indices]
        .copy()
        .reset_index(drop=True)
    )

    training_engines = set(
        training_dataframe[GROUP_COLUMN].unique()
    )

    validation_engines = set(
        validation_dataframe[GROUP_COLUMN].unique()
    )

    overlapping_engines = (
        training_engines.intersection(
            validation_engines
        )
    )

    if overlapping_engines:
        raise ValueError(
            "Engine leakage detected between training "
            "and validation datasets."
        )

    return training_dataframe, validation_dataframe


def create_models():
    models = {
        "Dummy Baseline": DummyRegressor(
            strategy="mean"
        ),

        "Linear Regression": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LinearRegression(),
                ),
            ]
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),

        "HistGradientBoosting": (
            HistGradientBoostingRegressor(
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


def calculate_metrics(
    actual_values,
    predicted_values,
):
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


def train_and_compare_models(
    training_dataframe,
    validation_dataframe,
    feature_columns,
):
    x_train = training_dataframe[
        feature_columns
    ]

    y_train = training_dataframe[
        TARGET_COLUMN
    ]

    x_validation = validation_dataframe[
        feature_columns
    ]

    y_validation = validation_dataframe[
        TARGET_COLUMN
    ]

    models = create_models()

    evaluation_records = []
    trained_models = {}
    validation_predictions = {}

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")

        start_time = time.perf_counter()

        model.fit(
            x_train,
            y_train,
        )

        training_duration = (
            time.perf_counter() - start_time
        )

        predicted_values = model.predict(
            x_validation
        )

        predicted_values = np.clip(
            predicted_values,
            0,
            125,
        )

        metrics = calculate_metrics(
            y_validation,
            predicted_values,
        )

        evaluation_records.append(
            {
                "model": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "training_seconds": round(
                    training_duration,
                    3,
                ),
            }
        )

        trained_models[model_name] = model

        validation_predictions[model_name] = (
            predicted_values
        )

        print(
            f"MAE: {metrics['mae']:.3f}"
        )

        print(
            f"RMSE: {metrics['rmse']:.3f}"
        )

        print(
            f"R-squared: {metrics['r2']:.3f}"
        )

    evaluation_dataframe = pd.DataFrame(
        evaluation_records
    )

    evaluation_dataframe = (
        evaluation_dataframe
        .sort_values(
            by="rmse",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    return (
        evaluation_dataframe,
        trained_models,
        validation_predictions,
    )


def evaluate_final_engine_cycles(
    validation_dataframe,
    feature_columns,
    trained_models,
):
    final_cycle_rows = (
        validation_dataframe
        .sort_values(
            ["engine_id", "cycle"]
        )
        .groupby("engine_id")
        .tail(1)
        .copy()
    )

    evaluation_records = []

    for model_name, model in trained_models.items():
        predicted_values = model.predict(
            final_cycle_rows[feature_columns]
        )

        predicted_values = np.clip(
            predicted_values,
            0,
            125,
        )

        metrics = calculate_metrics(
            final_cycle_rows[TARGET_COLUMN],
            predicted_values,
        )

        evaluation_records.append(
            {
                "model": model_name,
                "terminal_cycle_mae": metrics["mae"],
                "terminal_cycle_rmse": metrics["rmse"],
                "terminal_cycle_r2": metrics["r2"],
            }
        )

    result = pd.DataFrame(
        evaluation_records
    )

    result = result.sort_values(
        by="terminal_cycle_rmse",
        ascending=True,
    ).reset_index(drop=True)

    return result


def plot_model_comparison(
    evaluation_dataframe,
):
    plot_dataframe = evaluation_dataframe.melt(
        id_vars=["model"],
        value_vars=["mae", "rmse"],
        var_name="metric",
        value_name="error",
    )

    plt.figure(figsize=(11, 6))

    sns.barplot(
        data=plot_dataframe,
        x="model",
        y="error",
        hue="metric",
    )

    plt.title(
        "Grouped Validation Model Comparison"
    )

    plt.xlabel("Model")
    plt.ylabel("Error in RUL Cycles")
    plt.xticks(
        rotation=15,
        ha="right",
    )

    plt.legend(
        title="Metric"
    )

    plt.tight_layout()

    output_path = (
        EVALUATION_DIRECTORY
        / "model_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_actual_vs_predicted(
    validation_dataframe,
    predicted_values,
    model_name,
):
    actual_values = validation_dataframe[
        TARGET_COLUMN
    ].to_numpy()

    plt.figure(figsize=(8, 7))

    plt.scatter(
        actual_values,
        predicted_values,
        alpha=0.25,
        s=14,
    )

    minimum_value = min(
        actual_values.min(),
        predicted_values.min(),
    )

    maximum_value = max(
        actual_values.max(),
        predicted_values.max(),
    )

    plt.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Perfect prediction",
    )

    plt.title(
        f"Actual vs Predicted RUL: {model_name}"
    )

    plt.xlabel("Actual Capped RUL")
    plt.ylabel("Predicted Capped RUL")
    plt.legend()
    plt.tight_layout()

    output_path = (
        EVALUATION_DIRECTORY
        / "actual_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_residual_distribution(
    validation_dataframe,
    predicted_values,
    model_name,
):
    actual_values = validation_dataframe[
        TARGET_COLUMN
    ].to_numpy()

    residuals = (
        actual_values - predicted_values
    )

    plt.figure(figsize=(10, 6))

    sns.histplot(
        residuals,
        bins=35,
        kde=True,
    )

    plt.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    plt.title(
        f"Prediction Residual Distribution: {model_name}"
    )

    plt.xlabel(
        "Residual: Actual RUL - Predicted RUL"
    )

    plt.ylabel("Number of Observations")
    plt.tight_layout()

    output_path = (
        EVALUATION_DIRECTORY
        / "residual_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_validation_predictions(
    validation_dataframe,
    predicted_values,
    model_name,
):
    predictions_dataframe = (
        validation_dataframe[
            [
                "engine_id",
                "cycle",
                "rul",
                "rul_capped",
            ]
        ]
        .copy()
    )

    predictions_dataframe[
        "predicted_rul"
    ] = predicted_values

    predictions_dataframe[
        "residual"
    ] = (
        predictions_dataframe["rul_capped"]
        - predictions_dataframe["predicted_rul"]
    )

    predictions_dataframe[
        "model"
    ] = model_name

    output_path = (
        EVALUATION_DIRECTORY
        / "validation_predictions.csv"
    )

    predictions_dataframe.to_csv(
        output_path,
        index=False,
    )


def save_best_model(
    model,
    model_name,
    feature_columns,
    metrics,
    training_dataframe,
    validation_dataframe,
):
    model_path = (
        MODEL_DIRECTORY / "rul_model.joblib"
    )

    metadata_path = (
        MODEL_DIRECTORY / "rul_model_metadata.json"
    )

    feature_path = (
        MODEL_DIRECTORY / "rul_model_features.txt"
    )

    joblib.dump(
        model,
        model_path,
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
        "target": TARGET_COLUMN,
        "rul_cap": 125,
        "random_state": RANDOM_STATE,
        "validation_group": GROUP_COLUMN,
        "validation_group_fraction": VALIDATION_SIZE,
        "number_of_features": len(feature_columns),
        "training_engines": int(
            training_dataframe[
                "engine_id"
            ].nunique()
        ),
        "validation_engines": int(
            validation_dataframe[
                "engine_id"
            ].nunique()
        ),
        "validation_mae": float(
            metrics["mae"]
        ),
        "validation_rmse": float(
            metrics["rmse"]
        ),
        "validation_r2": float(
            metrics["r2"]
        ),
        "limitations": [
            "Trained on simulated NASA C-MAPSS FD001 data.",
            "Not validated for safety-critical operational use.",
            "Designed as a portfolio demonstration.",
            "Predictions are capped at 125 cycles.",
        ],
    }

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


def run_model_training():
    prepare_output_directories()

    dataframe = load_feature_dataframe()

    feature_columns = (
        load_model_feature_columns(
            dataframe
        )
    )

    (
        training_dataframe,
        validation_dataframe,
    ) = create_grouped_train_validation_split(
        dataframe,
        feature_columns,
    )

    training_engines = set(
        training_dataframe[
            "engine_id"
        ].unique()
    )

    validation_engines = set(
        validation_dataframe[
            "engine_id"
        ].unique()
    )

    print("\nFD001 BASELINE MODEL TRAINING")
    print("=" * 70)

    print(
        f"Training engines: "
        f"{len(training_engines)}"
    )

    print(
        f"Validation engines: "
        f"{len(validation_engines)}"
    )

    print(
        f"Overlapping engines: "
        f"{len(training_engines & validation_engines)}"
    )

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    (
        evaluation_dataframe,
        trained_models,
        validation_predictions,
    ) = train_and_compare_models(
        training_dataframe,
        validation_dataframe,
        feature_columns,
    )

    terminal_cycle_evaluation = (
        evaluate_final_engine_cycles(
            validation_dataframe,
            feature_columns,
            trained_models,
        )
    )

    best_model_name = (
        evaluation_dataframe.iloc[0]["model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    best_predictions = (
        validation_predictions[
            best_model_name
        ]
    )

    best_metrics = (
        evaluation_dataframe.iloc[0]
        .to_dict()
    )

    evaluation_dataframe.to_csv(
        EVALUATION_DIRECTORY
        / "model_comparison.csv",
        index=False,
    )

    terminal_cycle_evaluation.to_csv(
        EVALUATION_DIRECTORY
        / "terminal_cycle_evaluation.csv",
        index=False,
    )

    plot_model_comparison(
        evaluation_dataframe
    )

    plot_actual_vs_predicted(
        validation_dataframe,
        best_predictions,
        best_model_name,
    )

    plot_residual_distribution(
        validation_dataframe,
        best_predictions,
        best_model_name,
    )

    save_validation_predictions(
        validation_dataframe,
        best_predictions,
        best_model_name,
    )

    model_path = save_best_model(
        best_model,
        best_model_name,
        feature_columns,
        best_metrics,
        training_dataframe,
        validation_dataframe,
    )

    print("\nMODEL COMPARISON")
    print("=" * 70)

    print(
        evaluation_dataframe.to_string(
            index=False
        )
    )

    print("\nTERMINAL-CYCLE EVALUATION")
    print("=" * 70)

    print(
        terminal_cycle_evaluation.to_string(
            index=False
        )
    )

    print("\nBEST MODEL")
    print("=" * 70)

    print(
        f"Selected model: {best_model_name}"
    )

    print(
        f"Validation MAE: "
        f"{best_metrics['mae']:.3f}"
    )

    print(
        f"Validation RMSE: "
        f"{best_metrics['rmse']:.3f}"
    )

    print(
        f"Validation R-squared: "
        f"{best_metrics['r2']:.3f}"
    )

    print(
        f"Saved model: {model_path}"
    )


if __name__ == "__main__":
    run_model_training()