import pandas as pd
import pytest

from src.batch_prediction import (
    select_latest_asset_rows,
    validate_uploaded_data,
)
from src.data_loader import COLUMN_NAMES


def create_valid_upload_dataframe():
    records = []

    for engine_id in [
        1,
        2,
    ]:
        for cycle in range(
            1,
            4,
        ):
            record = {
                "engine_id": engine_id,
                "cycle": cycle,
                "operational_setting_1": 0.0,
                "operational_setting_2": 0.0,
                "operational_setting_3": 100.0,
            }

            for sensor_number in range(
                1,
                22,
            ):
                record[
                    f"sensor_{sensor_number}"
                ] = float(
                    sensor_number + cycle
                )

            records.append(record)

    return pd.DataFrame(
        records,
        columns=COLUMN_NAMES,
    )


def test_valid_upload_passes_validation():
    dataframe = (
        create_valid_upload_dataframe()
    )

    (
        cleaned_dataframe,
        warnings,
        summary,
    ) = validate_uploaded_data(
        dataframe
    )

    assert len(cleaned_dataframe) == 6
    assert summary["assets"] == 2
    assert summary["missing_values"] == 0


def test_missing_columns_fail_validation():
    dataframe = (
        create_valid_upload_dataframe()
        .drop(
            columns=["sensor_21"]
        )
    )

    with pytest.raises(
        ValueError
    ):
        validate_uploaded_data(
            dataframe
        )


def test_duplicate_engine_cycle_fails():
    dataframe = (
        create_valid_upload_dataframe()
    )

    duplicated_row = (
        dataframe.iloc[[0]]
    )

    dataframe = pd.concat(
        [
            dataframe,
            duplicated_row,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError
    ):
        validate_uploaded_data(
            dataframe
        )


def test_non_numeric_sensor_fails():
    dataframe = (
        create_valid_upload_dataframe()
    )

    dataframe.loc[
        0,
        "sensor_3",
    ] = "invalid"

    with pytest.raises(
        ValueError
    ):
        validate_uploaded_data(
            dataframe
        )


def test_latest_asset_rows_are_selected():
    dataframe = (
        create_valid_upload_dataframe()
    )

    latest_rows = (
        select_latest_asset_rows(
            dataframe
        )
    )

    assert len(latest_rows) == 2

    assert (
        latest_rows["cycle"] == 3
    ).all()