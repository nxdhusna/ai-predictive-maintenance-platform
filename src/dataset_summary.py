import pandas as pd

from data_loader import load_fd001_data
from preprocessing import add_training_rul


def create_dataset_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "metric": [
                "Number of observations",
                "Number of engines",
                "Minimum cycle",
                "Maximum cycle",
                "Average engine lifespan",
                "Minimum RUL",
                "Maximum RUL",
                "Missing values",
                "Duplicate rows",
            ],
            "value": [
                len(dataframe),
                dataframe["engine_id"].nunique(),
                dataframe["cycle"].min(),
                dataframe["cycle"].max(),
                round(
                    dataframe.groupby("engine_id")["cycle"]
                    .max()
                    .mean(),
                    2,
                ),
                dataframe["rul"].min(),
                dataframe["rul"].max(),
                dataframe.isna().sum().sum(),
                dataframe.duplicated().sum(),
            ],
        }
    )

    return summary


if __name__ == "__main__":
    train_df, _, _ = load_fd001_data()
    processed_train_df = add_training_rul(train_df)

    dataset_summary = create_dataset_summary(
        processed_train_df
    )

    print("\nFD001 TRAINING DATASET SUMMARY")
    print("=" * 50)
    print(dataset_summary.to_string(index=False))