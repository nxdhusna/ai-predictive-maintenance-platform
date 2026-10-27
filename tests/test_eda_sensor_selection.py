from src.data_loader import load_fd001_data
from src.preprocessing import add_training_rul
from src.eda_sensor_selection import (
    SENSOR_COLUMNS,
    calculate_feature_statistics,
    calculate_sensor_rul_correlations,
    create_sensor_selection_report,
)


def get_processed_training_data():
    train_dataframe, _, _ = load_fd001_data()

    return add_training_rul(
        train_dataframe
    )


def test_all_sensors_are_present():
    dataframe = get_processed_training_data()

    for sensor in SENSOR_COLUMNS:
        assert sensor in dataframe.columns


def test_feature_statistics_include_all_sensors():
    dataframe = get_processed_training_data()

    statistics = calculate_feature_statistics(
        dataframe
    )

    sensor_statistics = statistics[
        statistics["feature"].isin(
            SENSOR_COLUMNS
        )
    ]

    assert len(sensor_statistics) == 21


def test_sensor_correlations_include_all_sensors():
    dataframe = get_processed_training_data()

    correlations = calculate_sensor_rul_correlations(
        dataframe
    )

    assert len(correlations) == 21
    assert correlations["sensor"].nunique() == 21


def test_constant_sensors_are_dropped():
    dataframe = get_processed_training_data()

    statistics = calculate_feature_statistics(
        dataframe
    )

    correlations = calculate_sensor_rul_correlations(
        dataframe
    )

    report = create_sensor_selection_report(
        dataframe,
        statistics,
        correlations,
    )

    constant_sensors = report[
        report["variance"] <= 1e-8
    ]

    assert (
        constant_sensors["selection_status"] == "Drop"
    ).all()


def test_selected_sensors_have_variation():
    dataframe = get_processed_training_data()

    statistics = calculate_feature_statistics(
        dataframe
    )

    correlations = calculate_sensor_rul_correlations(
        dataframe
    )

    report = create_sensor_selection_report(
        dataframe,
        statistics,
        correlations,
    )

    selected_sensors = report[
        report["selection_status"] == "Keep"
    ]

    assert (
        selected_sensors["variance"] > 1e-8
    ).all()