from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.anomaly_detection import (
    assign_anomaly_severity,
    assign_anomaly_status,
    convert_to_anomaly_score,
)
from src.data_loader import COLUMN_NAMES
from src.feature_engineering import (
    ROLLING_WINDOWS,
    add_rolling_sensor_features,
    add_sensor_difference_features,
    load_selected_sensors,
)
from src.maintenance_intelligence import (
    assign_health_status,
    assign_maintenance_priority,
    assign_maintenance_window,
    assign_risk_level,
    calculate_health_score,
    calculate_risk_score,
    recommend_maintenance_action,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "rul_model.joblib"
)

RUL_FEATURE_PATH = (
    PROJECT_ROOT
    / "models"
    / "rul_model_features.txt"
)

FAILURE_CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "models"
    / "failure_classifier.joblib"
)

ANOMALY_DETECTOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "anomaly_detector.joblib"
)

RUL_CAP = 125


def load_text_feature_list(file_path):
    """
    Load an ordered text feature list.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Feature file was not found: {file_path}"
        )

    with file_path.open(
        mode="r",
        encoding="utf-8",
    ) as feature_file:
        features = [
            line.strip()
            for line in feature_file
            if line.strip()
        ]

    if not features:
        raise ValueError(
            f"Feature file is empty: {file_path}"
        )

    return features


def load_prediction_models():
    """
    Load all deployed predictive models.
    """

    required_model_paths = [
        RUL_MODEL_PATH,
        FAILURE_CLASSIFIER_PATH,
        ANOMALY_DETECTOR_PATH,
    ]

    missing_models = [
        str(path)
        for path in required_model_paths
        if not path.exists()
    ]

    if missing_models:
        raise FileNotFoundError(
            "Required model files are missing: "
            f"{missing_models}"
        )

    return {
        "rul_model": joblib.load(
            RUL_MODEL_PATH
        ),
        "rul_features": (
            load_text_feature_list(
                RUL_FEATURE_PATH
            )
        ),
        "failure_bundle": joblib.load(
            FAILURE_CLASSIFIER_PATH
        ),
        "anomaly_bundle": joblib.load(
            ANOMALY_DETECTOR_PATH
        ),
    }


def standardise_uploaded_columns(dataframe):
    """
    Clean column names without silently changing meaning.
    """

    result = dataframe.copy()

    result.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        for column in result.columns
    ]

    return result


def validate_uploaded_data(dataframe):
    """
    Validate the uploaded raw sensor-history data.

    Returns:
    - cleaned dataframe
    - warning messages
    - summary dictionary
    """

    if dataframe.empty:
        raise ValueError(
            "The uploaded CSV is empty."
        )

    result = standardise_uploaded_columns(
        dataframe
    )

    required_columns = set(
        COLUMN_NAMES
    )

    missing_columns = sorted(
        required_columns
        - set(result.columns)
    )

    if missing_columns:
        raise ValueError(
            "The uploaded file is missing required columns: "
            f"{missing_columns}"
        )

    result = result[COLUMN_NAMES].copy()

    numeric_columns = COLUMN_NAMES

    conversion_failures = {}

    for column in numeric_columns:
        original_non_null_count = int(
            result[column].notna().sum()
        )

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        converted_non_null_count = int(
            result[column].notna().sum()
        )

        failed_count = (
            original_non_null_count
            - converted_non_null_count
        )

        if failed_count > 0:
            conversion_failures[column] = (
                failed_count
            )

    if conversion_failures:
        raise ValueError(
            "Some values could not be converted to "
            f"numbers: {conversion_failures}"
        )

    missing_value_count = int(
        result.isna().sum().sum()
    )

    if missing_value_count > 0:
        missing_by_column = (
            result.isna()
            .sum()
            .loc[
                lambda values: values > 0
            ]
            .to_dict()
        )

        raise ValueError(
            "Missing sensor values were found: "
            f"{missing_by_column}"
        )

    if (
        result["engine_id"]
        % 1
        != 0
    ).any():
        raise ValueError(
            "Every engine_id must be a whole number."
        )

    if (
        result["cycle"]
        % 1
        != 0
    ).any():
        raise ValueError(
            "Every cycle must be a whole number."
        )

    result["engine_id"] = (
        result["engine_id"].astype(int)
    )

    result["cycle"] = (
        result["cycle"].astype(int)
    )

    if (
        result["engine_id"] <= 0
    ).any():
        raise ValueError(
            "Every engine_id must be greater than zero."
        )

    if (
        result["cycle"] <= 0
    ).any():
        raise ValueError(
            "Every cycle must be greater than zero."
        )

    duplicated_cycles = result.duplicated(
        subset=[
            "engine_id",
            "cycle",
        ]
    )

    duplicate_count = int(
        duplicated_cycles.sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            "The file contains "
            f"{duplicate_count} duplicated "
            "engine-cycle combinations."
        )

    result = (
        result
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .reset_index(drop=True)
    )

    warnings = []

    short_history_engines = (
        result.groupby("engine_id")
        .size()
        .loc[
            lambda values: values < max(
                ROLLING_WINDOWS
            )
        ]
    )

    if not short_history_engines.empty:
        warnings.append(
            "Some assets contain fewer than "
            f"{max(ROLLING_WINDOWS)} observations. "
            "Predictions can still be generated, but "
            "the longest rolling-window features will "
            "use limited history."
        )

    non_continuous_engines = []

    for engine_id, engine_data in (
        result.groupby("engine_id")
    ):
        cycles = (
            engine_data["cycle"]
            .sort_values()
            .to_numpy()
        )

        if len(cycles) > 1:
            cycle_differences = np.diff(
                cycles
            )

            if not (
                cycle_differences == 1
            ).all():
                non_continuous_engines.append(
                    int(engine_id)
                )

    if non_continuous_engines:
        warnings.append(
            "Non-continuous cycle histories were found "
            "for these assets: "
            f"{non_continuous_engines}. "
            "The model will use the available ordered rows."
        )

    summary = {
        "rows": int(
            len(result)
        ),
        "assets": int(
            result[
                "engine_id"
            ].nunique()
        ),
        "minimum_cycle": int(
            result["cycle"].min()
        ),
        "maximum_cycle": int(
            result["cycle"].max()
        ),
        "missing_values": 0,
        "duplicate_engine_cycles": 0,
    }

    return result, warnings, summary


def build_inference_features(
    raw_dataframe,
):
    """
    Create time-aware inference features without requiring
    an actual RUL target.

    Only present and past observations are used.
    """

    selected_sensors = (
        load_selected_sensors()
    )

    result = (
        raw_dataframe
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    result["log_cycle"] = np.log1p(
        result["cycle"]
    )

    result = (
        add_sensor_difference_features(
            result,
            selected_sensors,
        )
    )

    result = (
        add_rolling_sensor_features(
            result,
            selected_sensors,
            rolling_windows=(
                ROLLING_WINDOWS
            ),
        )
    )

    numeric_data = result.select_dtypes(
        include=[np.number]
    )

    if numeric_data.isna().any().any():
        raise ValueError(
            "Feature engineering generated "
            "missing values."
        )

    if np.isinf(
        numeric_data.to_numpy()
    ).any():
        raise ValueError(
            "Feature engineering generated "
            "infinite values."
        )

    return result


def validate_inference_features(
    feature_dataframe,
    required_features,
    model_name,
):
    """
    Confirm that every model input feature exists.
    """

    missing_features = [
        feature
        for feature in required_features
        if feature
        not in feature_dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"{model_name} is missing features: "
            f"{missing_features}"
        )


def select_latest_asset_rows(
    feature_dataframe,
):
    """
    Select the latest available cycle for each uploaded asset.
    """

    latest_rows = (
        feature_dataframe
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .groupby("engine_id")
        .tail(1)
        .sort_values(
            "engine_id"
        )
        .reset_index(drop=True)
    )

    return latest_rows


def predict_rul(
    latest_rows,
    rul_model,
    rul_features,
):
    """
    Generate capped RUL predictions.
    """

    validate_inference_features(
        latest_rows,
        rul_features,
        "RUL model",
    )

    predictions = rul_model.predict(
        latest_rows[
            rul_features
        ]
    )

    return np.clip(
        predictions,
        0,
        RUL_CAP,
    )


def predict_failure_probability(
    latest_rows,
    failure_bundle,
):
    """
    Generate calibrated failure-within-30-cycle probabilities.
    """

    feature_columns = failure_bundle[
        "feature_columns"
    ]

    validate_inference_features(
        latest_rows,
        feature_columns,
        "Failure classifier",
    )

    base_model = failure_bundle[
        "base_model"
    ]

    calibrator = failure_bundle[
        "probability_calibrator"
    ]

    raw_probabilities = (
        base_model.predict_proba(
            latest_rows[
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

    return np.clip(
        calibrated_probabilities,
        0,
        1,
    )


def predict_anomalies(
    latest_rows,
    anomaly_bundle,
):
    """
    Generate anomaly scores and labels.
    """

    feature_columns = anomaly_bundle[
        "feature_columns"
    ]

    validate_inference_features(
        latest_rows,
        feature_columns,
        "Anomaly detector",
    )

    scaler = anomaly_bundle[
        "scaler"
    ]

    detector = anomaly_bundle[
        "detector"
    ]

    healthy_reference_scores = (
        anomaly_bundle[
            "healthy_reference_scores"
        ]
    )

    scaled_features = scaler.transform(
        latest_rows[
            feature_columns
        ]
    )

    decision_scores = (
        detector.decision_function(
            scaled_features
        )
    )

    detector_predictions = (
        detector.predict(
            scaled_features
        )
    )

    anomaly_scores = (
        convert_to_anomaly_score(
            decision_scores,
            healthy_reference_scores,
        )
    )

    return {
        "decision_scores": (
            decision_scores
        ),
        "detector_predictions": (
            detector_predictions
        ),
        "anomaly_scores": (
            anomaly_scores
        ),
    }


def create_batch_predictions(
    raw_dataframe,
    model_bundle,
):
    """
    Run all prediction and decision-support components.
    """

    feature_dataframe = (
        build_inference_features(
            raw_dataframe
        )
    )

    latest_rows = select_latest_asset_rows(
        feature_dataframe
    )

    predicted_rul = predict_rul(
        latest_rows,
        model_bundle["rul_model"],
        model_bundle["rul_features"],
    )

    failure_probabilities = (
        predict_failure_probability(
            latest_rows,
            model_bundle[
                "failure_bundle"
            ],
        )
    )

    anomaly_results = predict_anomalies(
        latest_rows,
        model_bundle[
            "anomaly_bundle"
        ],
    )

    output = latest_rows[
        [
            "engine_id",
            "cycle",
        ]
    ].copy()

    output = output.rename(
        columns={
            "cycle": (
                "latest_observed_cycle"
            ),
        }
    )

    output["predicted_rul"] = (
        predicted_rul
    )

    output[
        "failure_within_30_cycles_probability"
    ] = failure_probabilities

    output[
        "failure_probability_percent"
    ] = (
        failure_probabilities
        * 100
    ).round(1)

    output["health_score"] = (
        calculate_health_score(
            predicted_rul
        )
    )

    output["risk_score"] = (
        calculate_risk_score(
            output[
                "health_score"
            ]
        )
    )

    output["health_status"] = (
        output["health_score"]
        .apply(
            assign_health_status
        )
    )

    output["risk_level"] = (
        output["risk_score"]
        .apply(
            assign_risk_level
        )
    )

    output[
        "maintenance_priority"
    ] = (
        output["predicted_rul"]
        .apply(
            assign_maintenance_priority
        )
    )

    output[
        "maintenance_window"
    ] = (
        output["predicted_rul"]
        .apply(
            assign_maintenance_window
        )
    )

    output[
        "recommended_action"
    ] = (
        output["predicted_rul"]
        .apply(
            recommend_maintenance_action
        )
    )

    output[
        "anomaly_decision_score"
    ] = anomaly_results[
        "decision_scores"
    ]

    output["anomaly_score"] = (
        anomaly_results[
            "anomaly_scores"
        ]
    )

    output["anomaly_status"] = [
        assign_anomaly_status(
            prediction
        )
        for prediction in anomaly_results[
            "detector_predictions"
        ]
    ]

    output["anomaly_severity"] = (
        output["anomaly_score"]
        .apply(
            assign_anomaly_severity
        )
    )

    output["history_rows_used"] = (
        output["engine_id"]
        .map(
            raw_dataframe.groupby(
                "engine_id"
            ).size()
        )
    )

    output["model_scope_note"] = (
        "Portfolio prediction generated from "
        "simulated-data models."
    )

    priority_mapping = {
        "P1 - Immediate": 1,
        "P2 - High": 2,
        "P3 - Scheduled": 3,
        "P4 - Routine": 4,
    }

    output["_priority_number"] = (
        output[
            "maintenance_priority"
        ]
        .map(priority_mapping)
    )

    output = (
        output
        .sort_values(
            by=[
                "_priority_number",
                "predicted_rul",
                "anomaly_score",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .drop(
            columns=[
                "_priority_number"
            ]
        )
        .reset_index(drop=True)
    )

    output[
        "batch_priority_rank"
    ] = range(
        1,
        len(output) + 1,
    )

    first_columns = [
        "batch_priority_rank",
        "engine_id",
        "latest_observed_cycle",
    ]

    remaining_columns = [
        column
        for column in output.columns
        if column not in first_columns
    ]

    output = output[
        first_columns
        + remaining_columns
    ]

    return output, feature_dataframe


def create_batch_summary(
    prediction_dataframe,
):
    """
    Create high-level batch prediction KPIs.
    """

    return {
        "assets_scored": int(
            len(prediction_dataframe)
        ),
        "critical_assets": int(
            (
                prediction_dataframe[
                    "risk_level"
                ]
                == "Critical"
            ).sum()
        ),
        "p1_assets": int(
            (
                prediction_dataframe[
                    "maintenance_priority"
                ]
                == "P1 - Immediate"
            ).sum()
        ),
        "anomalous_assets": int(
            (
                prediction_dataframe[
                    "anomaly_status"
                ]
                == "Anomaly Detected"
            ).sum()
        ),
        "average_predicted_rul": float(
            prediction_dataframe[
                "predicted_rul"
            ].mean()
        ),
        "average_failure_probability": float(
            prediction_dataframe[
                "failure_within_30_cycles_probability"
            ].mean()
        ),
    }