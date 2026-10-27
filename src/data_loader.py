from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


COLUMN_NAMES = [
    "engine_id",
    "cycle",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


def load_sensor_file(file_path: Path) -> pd.DataFrame:
    """
    Load a C-MAPSS sensor-data file.

    Each row contains:
    - engine identifier
    - operational cycle
    - three operational settings
    - twenty-one sensor measurements
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {file_path}"
        )

    dataframe = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        engine="python",
    )

    return dataframe


def load_rul_file(file_path: Path) -> pd.DataFrame:
    """
    Load the true Remaining Useful Life values for test engines.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"RUL file was not found: {file_path}"
        )

    rul_dataframe = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["additional_rul"],
        engine="python",
    )

    rul_dataframe.insert(
        loc=0,
        column="engine_id",
        value=range(1, len(rul_dataframe) + 1),
    )

    return rul_dataframe


def load_fd001_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the complete FD001 training, test, and test-RUL data.
    """

    train_path = RAW_DATA_DIR / "train_FD001.txt"
    test_path = RAW_DATA_DIR / "test_FD001.txt"
    rul_path = RAW_DATA_DIR / "RUL_FD001.txt"

    train_dataframe = load_sensor_file(train_path)
    test_dataframe = load_sensor_file(test_path)
    rul_dataframe = load_rul_file(rul_path)

    return train_dataframe, test_dataframe, rul_dataframe


if __name__ == "__main__":
    train_df, test_df, rul_df = load_fd001_data()

    print("FD001 data loaded successfully.")
    print(f"Training shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"RUL shape: {rul_df.shape}")

    print("\nTraining data preview:")
    print(train_df.head())

    print("\nTest RUL preview:")
    print(rul_df.head())