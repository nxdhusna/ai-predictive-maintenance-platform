from pathlib import Path

from src.data_loader import (
    RAW_DATA_DIR,
    load_sensor_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "samples"
)

SAMPLE_OUTPUT_PATH = (
    SAMPLE_DIRECTORY
    / "sample_batch_sensor_history.csv"
)


def create_batch_sample():
    """
    Create a small sample upload containing the complete
    observed history of three FD001 test engines.
    """

    source_path = (
        RAW_DATA_DIR
        / "test_FD001.txt"
    )

    test_dataframe = load_sensor_file(
        source_path
    )

    sample_dataframe = (
        test_dataframe.loc[
            test_dataframe[
                "engine_id"
            ].isin(
                [1, 2, 3]
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    SAMPLE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_dataframe.to_csv(
        SAMPLE_OUTPUT_PATH,
        index=False,
    )

    print(
        "Sample batch file created:"
    )

    print(SAMPLE_OUTPUT_PATH)

    print(
        f"Rows: {len(sample_dataframe)}"
    )

    print(
        "Assets: "
        f"{sample_dataframe['engine_id'].nunique()}"
    )


if __name__ == "__main__":
    create_batch_sample()