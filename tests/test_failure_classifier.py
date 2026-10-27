import numpy as np
import pandas as pd

from src.failure_classifier import (
    TARGET_COLUMN,
    add_failure_target,
    calculate_classification_metrics,
    split_engine_groups,
)


def create_classifier_test_dataframe():
    records = []

    for engine_id in range(1, 21):
        for cycle in range(1, 11):
            rul = 10 - cycle

            records.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "rul": rul,
                    "sensor_2": float(
                        engine_id + cycle
                    ),
                }
            )

    return pd.DataFrame(records)


def test_failure_target_is_positive_within_horizon():
    dataframe = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1],
            "cycle": [1, 2, 3, 4],
            "rul": [50, 31, 30, 5],
        }
    )

    result = add_failure_target(
        dataframe,
        failure_horizon=30,
    )

    assert result[TARGET_COLUMN].tolist() == [
        0,
        0,
        1,
        1,
    ]


def test_engine_splits_do_not_overlap():
    dataframe = create_classifier_test_dataframe()

    (
        training_dataframe,
        calibration_dataframe,
        validation_dataframe,
    ) = split_engine_groups(
        dataframe
    )

    training_engines = set(
        training_dataframe[
            "engine_id"
        ].unique()
    )

    calibration_engines = set(
        calibration_dataframe[
            "engine_id"
        ].unique()
    )

    validation_engines = set(
        validation_dataframe[
            "engine_id"
        ].unique()
    )

    assert training_engines.isdisjoint(
        calibration_engines
    )

    assert training_engines.isdisjoint(
        validation_engines
    )

    assert calibration_engines.isdisjoint(
        validation_engines
    )


def test_engine_splits_preserve_all_rows():
    dataframe = create_classifier_test_dataframe()

    (
        training_dataframe,
        calibration_dataframe,
        validation_dataframe,
    ) = split_engine_groups(
        dataframe
    )

    total_rows = (
        len(training_dataframe)
        + len(calibration_dataframe)
        + len(validation_dataframe)
    )

    assert total_rows == len(dataframe)


def test_perfect_classification_metrics():
    actual_values = np.array(
        [0, 0, 1, 1]
    )

    predicted_labels = np.array(
        [0, 0, 1, 1]
    )

    predicted_probabilities = np.array(
        [0.05, 0.10, 0.90, 0.95]
    )

    metrics = calculate_classification_metrics(
        actual_values,
        predicted_labels,
        predicted_probabilities,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0