import numpy as np
import pandas as pd

from src.feature_engineering import (
    RUL_CAP,
    add_capped_rul,
    add_lifecycle_features,
    add_rolling_sensor_features,
    add_sensor_difference_features,
)


def create_small_test_dataframe():
    return pd.DataFrame(
        {
            "engine_id": [
                1, 1, 1,
                2, 2, 2,
            ],
            "cycle": [
                1, 2, 3,
                1, 2, 3,
            ],
            "sensor_2": [
                10.0, 12.0, 14.0,
                100.0, 103.0, 109.0,
            ],
            "rul": [
                200, 199, 198,
                2, 1, 0,
            ],
        }
    )


def test_capped_rul_does_not_exceed_cap():
    dataframe = create_small_test_dataframe()

    result = add_capped_rul(dataframe)

    assert result["rul_capped"].max() == RUL_CAP
    assert result["rul_capped"].min() == 0


def test_difference_resets_at_engine_boundary():
    dataframe = create_small_test_dataframe()

    result = add_sensor_difference_features(
        dataframe,
        ["sensor_2"],
    )

    engine_one_first_difference = result.loc[
        result["engine_id"] == 1,
        "sensor_2_diff_1",
    ].iloc[0]

    engine_two_first_difference = result.loc[
        result["engine_id"] == 2,
        "sensor_2_diff_1",
    ].iloc[0]

    assert engine_one_first_difference == 0.0
    assert engine_two_first_difference == 0.0


def test_difference_uses_previous_cycle():
    dataframe = create_small_test_dataframe()

    result = add_sensor_difference_features(
        dataframe,
        ["sensor_2"],
    )

    engine_one_differences = result.loc[
        result["engine_id"] == 1,
        "sensor_2_diff_1",
    ].tolist()

    assert engine_one_differences == [
        0.0,
        2.0,
        2.0,
    ]


def test_rolling_mean_does_not_cross_engines():
    dataframe = create_small_test_dataframe()

    result = add_rolling_sensor_features(
        dataframe,
        ["sensor_2"],
        rolling_windows=[2],
    )

    engine_two_first_mean = result.loc[
        result["engine_id"] == 2,
        "sensor_2_rolling_mean_2",
    ].iloc[0]

    assert engine_two_first_mean == 100.0


def test_rolling_mean_uses_current_and_past():
    dataframe = create_small_test_dataframe()

    result = add_rolling_sensor_features(
        dataframe,
        ["sensor_2"],
        rolling_windows=[2],
    )

    engine_one_means = result.loc[
        result["engine_id"] == 1,
        "sensor_2_rolling_mean_2",
    ].tolist()

    assert np.allclose(
        engine_one_means,
        [10.0, 11.0, 13.0],
    )


def test_lifecycle_fraction_is_bounded():
    dataframe = create_small_test_dataframe()

    result = add_lifecycle_features(
        dataframe
    )

    assert (
        result["observed_life_fraction"] > 0
    ).all()

    assert (
        result["observed_life_fraction"] <= 1
    ).all()