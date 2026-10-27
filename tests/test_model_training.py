import pandas as pd

from src.model_training import (
    calculate_metrics,
    create_grouped_train_validation_split,
)


def create_grouped_test_dataframe():
    records = []

    for engine_id in range(1, 11):
        for cycle in range(1, 6):
            records.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "sensor_2": float(
                        engine_id + cycle
                    ),
                    "rul_capped": float(
                        5 - cycle
                    ),
                }
            )

    return pd.DataFrame(records)


def test_grouped_split_has_no_engine_overlap():
    dataframe = create_grouped_test_dataframe()

    training_dataframe, validation_dataframe = (
        create_grouped_train_validation_split(
            dataframe,
            ["cycle", "sensor_2"],
        )
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

    assert training_engines.isdisjoint(
        validation_engines
    )


def test_grouped_split_preserves_all_rows():
    dataframe = create_grouped_test_dataframe()

    training_dataframe, validation_dataframe = (
        create_grouped_train_validation_split(
            dataframe,
            ["cycle", "sensor_2"],
        )
    )

    total_split_rows = (
        len(training_dataframe)
        + len(validation_dataframe)
    )

    assert total_split_rows == len(dataframe)


def test_metrics_are_zero_for_perfect_predictions():
    actual_values = [10, 20, 30]
    predicted_values = [10, 20, 30]

    metrics = calculate_metrics(
        actual_values,
        predicted_values,
    )

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0