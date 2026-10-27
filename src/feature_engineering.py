from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    PROCESSED_DATA_DIR,
    load_fd001_data,
)
from src.preprocessing import (
    add_test_rul,
    add_training_rul,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

SELECTED_SENSOR_FILE = (
    EDA_OUTPUT_DIR / "selected_sensor_features.txt"
)

RUL_CAP = 125

ROLLING_WINDOWS = [5, 10, 20]


def load_selected_sensors():
    """
    Load the sensor list produced during the EDA stage.
    """

    if not SELECTED_SENSOR_FILE.exists():
        raise FileNotFoundError(
            "Selected sensor file was not found. "
            "Run 'python -m src.eda_sensor_selection' first."
        )

    with SELECTED_SENSOR_FILE.open(
        mode="r",
        encoding="utf-8",
    ) as sensor_file:
        selected_sensors = [
            line.strip()
            for line in sensor_file
            if line.strip()
        ]

    if not selected_sensors:
        raise ValueError(
            "The selected sensor file is empty."
        )

    return selected_sensors


def add_capped_rul(
    dataframe,
    rul_cap=RUL_CAP,
):
    """
    Add a capped RUL target.

    Early engine life is treated as a relatively stable healthy
    period. RUL values above the selected cap are reduced to the cap.
    """

    result = dataframe.copy()

    if "rul" not in result.columns:
        raise ValueError(
            "The dataframe must contain a 'rul' column."
        )

    result["rul_capped"] = result["rul"].clip(
        upper=rul_cap
    )

    return result


def add_lifecycle_features(dataframe):
    """
    Add features describing the engine's observed lifecycle progress.
    """

    result = dataframe.copy()

    maximum_observed_cycle = (
        result.groupby("engine_id")["cycle"]
        .transform("max")
    )

    result["observed_life_fraction"] = (
        result["cycle"] / maximum_observed_cycle
    )

    result["log_cycle"] = np.log1p(
        result["cycle"]
    )

    return result


def add_sensor_difference_features(
    dataframe,
    selected_sensors,
):
    """
    Add previous-cycle sensor differences.

    Difference values are calculated separately for each engine,
    preventing values from one engine from affecting another.
    """

    result = dataframe.copy()

    grouped_data = result.groupby(
        "engine_id",
        sort=False,
    )

    for sensor in selected_sensors:
        difference_column = f"{sensor}_diff_1"

        result[difference_column] = (
            grouped_data[sensor]
            .diff()
            .fillna(0.0)
        )

    return result


def add_rolling_sensor_features(
    dataframe,
    selected_sensors,
    rolling_windows=ROLLING_WINDOWS,
):
    """
    Create rolling mean and standard deviation features.

    Rolling calculations use only the current and previous rows
    belonging to the same engine.
    """

    result = dataframe.copy()

    result = result.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    for sensor in selected_sensors:
        grouped_sensor = result.groupby(
            "engine_id",
            sort=False,
        )[sensor]

        for window in rolling_windows:
            mean_column = (
                f"{sensor}_rolling_mean_{window}"
            )

            standard_deviation_column = (
                f"{sensor}_rolling_std_{window}"
            )

            result[mean_column] = (
                grouped_sensor
                .rolling(
                    window=window,
                    min_periods=1,
                )
                .mean()
                .reset_index(
                    level=0,
                    drop=True,
                )
            )

            result[standard_deviation_column] = (
                grouped_sensor
                .rolling(
                    window=window,
                    min_periods=2,
                )
                .std()
                .reset_index(
                    level=0,
                    drop=True,
                )
                .fillna(0.0)
            )

    return result


def build_feature_dataframe(
    dataframe,
    selected_sensors,
):
    """
    Build the complete time-aware feature dataframe.
    """

    result = dataframe.copy()

    result = result.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

    result = add_capped_rul(result)

    result = add_lifecycle_features(result)

    result = add_sensor_difference_features(
        result,
        selected_sensors,
    )

    result = add_rolling_sensor_features(
        result,
        selected_sensors,
    )

    return result


def get_model_feature_columns(
    feature_dataframe,
    selected_sensors,
):
    """
    Return feature columns that may be used by the model.

    Engine ID and target columns are intentionally excluded.
    """

    engineered_feature_columns = []

    for column in feature_dataframe.columns:
        if (
            "_diff_" in column
            or "_rolling_mean_" in column
            or "_rolling_std_" in column
        ):
            engineered_feature_columns.append(column)

    base_feature_columns = [
        "cycle",
        "operational_setting_1",
        "operational_setting_2",
        "operational_setting_3",
        "observed_life_fraction",
        "log_cycle",
    ]

    model_feature_columns = (
        base_feature_columns
        + selected_sensors
        + engineered_feature_columns
    )

    model_feature_columns = [
        column
        for column in model_feature_columns
        if column in feature_dataframe.columns
    ]

    return model_feature_columns


def validate_engine_boundaries(
    original_dataframe,
    feature_dataframe,
    selected_sensors,
):
    """
    Confirm that the first cycle of every engine has zero
    difference features.

    This helps detect accidental calculations across engine
    boundaries.
    """

    first_engine_rows = (
        feature_dataframe
        .sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .head(1)
    )

    difference_columns = [
        f"{sensor}_diff_1"
        for sensor in selected_sensors
    ]

    invalid_difference_count = int(
        (
            first_engine_rows[difference_columns]
            .abs()
            .sum(axis=1)
            > 0
        ).sum()
    )

    original_engine_count = (
        original_dataframe["engine_id"].nunique()
    )

    feature_engine_count = (
        feature_dataframe["engine_id"].nunique()
    )

    if invalid_difference_count != 0:
        raise ValueError(
            "Feature leakage detected across engine boundaries."
        )

    if original_engine_count != feature_engine_count:
        raise ValueError(
            "Engine count changed during feature engineering."
        )


def validate_feature_dataframe(feature_dataframe):
    """
    Validate the final feature-engineered dataset.
    """

    missing_value_count = int(
        feature_dataframe.isna().sum().sum()
    )

    infinite_value_count = int(
        np.isinf(
            feature_dataframe.select_dtypes(
                include=[np.number]
            )
        ).sum().sum()
    )

    negative_capped_rul_count = int(
        (
            feature_dataframe["rul_capped"] < 0
        ).sum()
    )

    excessive_capped_rul_count = int(
        (
            feature_dataframe["rul_capped"] > RUL_CAP
        ).sum()
    )

    if missing_value_count != 0:
        raise ValueError(
            f"Feature dataframe contains "
            f"{missing_value_count} missing values."
        )

    if infinite_value_count != 0:
        raise ValueError(
            f"Feature dataframe contains "
            f"{infinite_value_count} infinite values."
        )

    if negative_capped_rul_count != 0:
        raise ValueError(
            "Feature dataframe contains negative capped RUL values."
        )

    if excessive_capped_rul_count != 0:
        raise ValueError(
            f"Some capped RUL values exceed {RUL_CAP}."
        )


def save_feature_data(
    dataframe,
    output_filename,
):
    """
    Save the feature-engineered dataframe.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DATA_DIR / output_filename
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_model_feature_list(feature_columns):
    """
    Save the exact model-feature list for reproducibility.
    """

    output_path = (
        EDA_OUTPUT_DIR / "model_feature_columns.txt"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        for feature in feature_columns:
            output_file.write(f"{feature}\n")

    return output_path


def run_feature_engineering():
    """
    Execute feature engineering for FD001 train and test data.
    """

    selected_sensors = load_selected_sensors()

    train_dataframe, test_dataframe, rul_dataframe = (
        load_fd001_data()
    )

    training_with_rul = add_training_rul(
        train_dataframe
    )

    testing_with_rul = add_test_rul(
        test_dataframe,
        rul_dataframe,
    )

    train_feature_dataframe = (
        build_feature_dataframe(
            training_with_rul,
            selected_sensors,
        )
    )

    test_feature_dataframe = (
        build_feature_dataframe(
            testing_with_rul,
            selected_sensors,
        )
    )

    validate_engine_boundaries(
        training_with_rul,
        train_feature_dataframe,
        selected_sensors,
    )

    validate_engine_boundaries(
        testing_with_rul,
        test_feature_dataframe,
        selected_sensors,
    )

    validate_feature_dataframe(
        train_feature_dataframe
    )

    validate_feature_dataframe(
        test_feature_dataframe
    )

    model_feature_columns = (
        get_model_feature_columns(
            train_feature_dataframe,
            selected_sensors,
        )
    )

    train_output_path = save_feature_data(
        train_feature_dataframe,
        "train_FD001_features.csv",
    )

    test_output_path = save_feature_data(
        test_feature_dataframe,
        "test_FD001_features.csv",
    )

    feature_list_path = save_model_feature_list(
        model_feature_columns
    )

    print("\nFD001 FEATURE ENGINEERING")
    print("=" * 70)

    print(
        f"Selected raw sensors: "
        f"{len(selected_sensors)}"
    )

    print(
        f"Total model features: "
        f"{len(model_feature_columns)}"
    )

    print(
        f"Training feature shape: "
        f"{train_feature_dataframe.shape}"
    )

    print(
        f"Test feature shape: "
        f"{test_feature_dataframe.shape}"
    )

    print(
        f"Maximum capped RUL: "
        f"{train_feature_dataframe['rul_capped'].max()}"
    )

    print(
        f"Missing training values: "
        f"{train_feature_dataframe.isna().sum().sum()}"
    )

    print(
        f"Training features saved to: "
        f"{train_output_path}"
    )

    print(
        f"Test features saved to: "
        f"{test_output_path}"
    )

    print(
        f"Model feature list saved to: "
        f"{feature_list_path}"
    )


if __name__ == "__main__":
    run_feature_engineering()