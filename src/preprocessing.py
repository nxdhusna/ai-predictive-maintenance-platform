from pathlib import Path

import pandas as pd

from src.data_loader import (
    PROCESSED_DATA_DIR,
    load_fd001_data,
)


def add_training_rul(
    train_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life for each training observation.

    Training engines are observed until failure. Therefore:

    RUL = final cycle for the engine - current cycle
    """

    dataframe = train_dataframe.copy()

    maximum_cycles = (
        dataframe.groupby("engine_id")["cycle"]
        .max()
        .rename("maximum_cycle")
        .reset_index()
    )

    dataframe = dataframe.merge(
        maximum_cycles,
        on="engine_id",
        how="left",
        validate="many_to_one",
    )

    dataframe["rul"] = (
        dataframe["maximum_cycle"] - dataframe["cycle"]
    )

    dataframe = dataframe.drop(columns=["maximum_cycle"])

    return dataframe


def add_test_rul(
    test_dataframe: pd.DataFrame,
    rul_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the true RUL for each test observation.

    NASA provides the additional RUL after each engine's final
    observed test cycle.
    """

    dataframe = test_dataframe.copy()

    final_observed_cycles = (
        dataframe.groupby("engine_id")["cycle"]
        .max()
        .rename("final_observed_cycle")
        .reset_index()
    )

    engine_rul = final_observed_cycles.merge(
        rul_dataframe,
        on="engine_id",
        how="left",
        validate="one_to_one",
    )

    dataframe = dataframe.merge(
        engine_rul,
        on="engine_id",
        how="left",
        validate="many_to_one",
    )

    dataframe["rul"] = (
        dataframe["final_observed_cycle"]
        + dataframe["additional_rul"]
        - dataframe["cycle"]
    )

    dataframe = dataframe.drop(
        columns=[
            "final_observed_cycle",
            "additional_rul",
        ]
    )

    return dataframe


def save_processed_data(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )


if __name__ == "__main__":
    train_df, test_df, test_rul_df = load_fd001_data()

    processed_train_df = add_training_rul(train_df)
    processed_test_df = add_test_rul(
        test_df,
        test_rul_df,
    )

    train_output = (
        PROCESSED_DATA_DIR / "train_FD001_processed.csv"
    )

    test_output = (
        PROCESSED_DATA_DIR / "test_FD001_processed.csv"
    )

    save_processed_data(
        processed_train_df,
        train_output,
    )

    save_processed_data(
        processed_test_df,
        test_output,
    )

    print("Processed datasets created successfully.")
    print(f"Training output: {train_output}")
    print(f"Test output: {test_output}")

    print("\nProcessed training preview:")
    print(
        processed_train_df[
            ["engine_id", "cycle", "rul"]
        ].head()
    )

    print("\nEngine 1 final training cycles:")
    print(
        processed_train_df.loc[
            processed_train_df["engine_id"] == 1,
            ["engine_id", "cycle", "rul"],
        ].tail()
    )