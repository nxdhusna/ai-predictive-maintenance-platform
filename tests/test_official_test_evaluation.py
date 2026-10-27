import numpy as np
import pandas as pd

from src.official_test_evaluation import (
    calculate_metrics,
    get_final_engine_cycles,
)


def create_test_engine_dataframe():
    return pd.DataFrame(
        {
            "engine_id": [
                1, 1, 1,
                2, 2,
                3, 3, 3, 3,
            ],
            "cycle": [
                1, 2, 3,
                1, 2,
                1, 2, 3, 4,
            ],
            "rul": [
                7, 6, 5,
                9, 8,
                6, 5, 4, 3,
            ],
            "rul_capped": [
                7, 6, 5,
                9, 8,
                6, 5, 4, 3,
            ],
        }
    )


def test_final_engine_cycles_selects_one_row_per_engine():
    dataframe = create_test_engine_dataframe()

    final_rows = get_final_engine_cycles(
        dataframe,
        expected_engine_count=3,
    )

    assert len(final_rows) == 3
    assert final_rows["engine_id"].nunique() == 3


def test_final_engine_cycles_are_correct():
    dataframe = create_test_engine_dataframe()

    final_rows = get_final_engine_cycles(
        dataframe
    )

    final_cycles = dict(
        zip(
            final_rows["engine_id"],
            final_rows["cycle"],
        )
    )

    assert final_cycles == {
        1: 3,
        2: 2,
        3: 4,
    }


def test_calculate_metrics_for_perfect_predictions():
    actual_values = np.array(
        [10.0, 20.0, 30.0]
    )

    predicted_values = np.array(
        [10.0, 20.0, 30.0]
    )

    metrics = calculate_metrics(
        actual_values,
        predicted_values,
    )

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0