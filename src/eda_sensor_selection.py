from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import load_fd001_data
from src.preprocessing import add_training_rul


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

SENSOR_COLUMNS = [
    f"sensor_{sensor_number}"
    for sensor_number in range(1, 22)
]

OPERATIONAL_SETTING_COLUMNS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]

VARIANCE_THRESHOLD = 1e-8


def prepare_output_directory() -> None:
    """
    Create the EDA output directory when it does not exist.
    """

    EDA_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_dataset_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a high-level summary of the FD001 training dataset.
    """

    engine_lifespans = (
        dataframe.groupby("engine_id")["cycle"].max()
    )

    summary_data = {
        "metric": [
            "Number of observations",
            "Number of engines",
            "Number of sensors",
            "Minimum engine lifespan",
            "Maximum engine lifespan",
            "Average engine lifespan",
            "Median engine lifespan",
            "Minimum RUL",
            "Maximum RUL",
            "Missing values",
            "Duplicate rows",
        ],
        "value": [
            len(dataframe),
            dataframe["engine_id"].nunique(),
            len(SENSOR_COLUMNS),
            int(engine_lifespans.min()),
            int(engine_lifespans.max()),
            round(engine_lifespans.mean(), 2),
            round(engine_lifespans.median(), 2),
            int(dataframe["rul"].min()),
            int(dataframe["rul"].max()),
            int(dataframe.isna().sum().sum()),
            int(dataframe.duplicated().sum()),
        ],
    }

    return pd.DataFrame(summary_data)


def calculate_feature_statistics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate descriptive statistics for operational settings
    and sensor measurements.
    """

    feature_columns = (
        OPERATIONAL_SETTING_COLUMNS
        + SENSOR_COLUMNS
    )

    statistics = (
        dataframe[feature_columns]
        .describe()
        .transpose()
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    statistics["variance"] = (
        dataframe[feature_columns]
        .var()
        .values
    )

    statistics["unique_values"] = [
        dataframe[column].nunique()
        for column in feature_columns
    ]

    statistics["missing_values"] = [
        dataframe[column].isna().sum()
        for column in feature_columns
    ]

    statistics["is_constant_or_near_constant"] = (
        statistics["variance"] <= VARIANCE_THRESHOLD
    )

    return statistics


def calculate_sensor_rul_correlations(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Pearson and Spearman correlations between
    each sensor and Remaining Useful Life.

    Spearman correlation is useful for detecting monotonic
    relationships that are not necessarily linear.
    """

    correlation_records = []

    for sensor_column in SENSOR_COLUMNS:
        pearson_correlation = dataframe[
            sensor_column
        ].corr(
            dataframe["rul"],
            method="pearson",
        )

        spearman_correlation = dataframe[
            sensor_column
        ].corr(
            dataframe["rul"],
            method="spearman",
        )

        correlation_records.append(
            {
                "sensor": sensor_column,
                "pearson_rul_correlation": pearson_correlation,
                "spearman_rul_correlation": spearman_correlation,
                "absolute_spearman_correlation": abs(
                    spearman_correlation
                ),
            }
        )

    correlation_dataframe = pd.DataFrame(
        correlation_records
    )

    correlation_dataframe = (
        correlation_dataframe
        .sort_values(
            by="absolute_spearman_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return correlation_dataframe


def create_sensor_selection_report(
    dataframe: pd.DataFrame,
    feature_statistics: pd.DataFrame,
    correlation_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decide which sensors should be retained.

    Current selection criteria:
    1. Remove constant or near-constant sensors.
    2. Retain sensors that provide measurable variation.
    3. Record RUL correlation for ranking and interpretation.

    Correlation is not used as the only exclusion criterion
    because a sensor can be useful to a nonlinear model even
    when its individual linear correlation is weak.
    """

    sensor_statistics = feature_statistics[
        feature_statistics["feature"].isin(
            SENSOR_COLUMNS
        )
    ].copy()

    sensor_statistics = sensor_statistics.rename(
        columns={"feature": "sensor"}
    )

    selection_report = sensor_statistics.merge(
        correlation_dataframe,
        on="sensor",
        how="left",
        validate="one_to_one",
    )

    selection_report["selection_status"] = (
        selection_report[
            "is_constant_or_near_constant"
        ].map(
            {
                True: "Drop",
                False: "Keep",
            }
        )
    )

    selection_report["selection_reason"] = (
        selection_report[
            "is_constant_or_near_constant"
        ].map(
            {
                True: (
                    "Constant or near-constant in FD001; "
                    "provides little predictive information."
                ),
                False: (
                    "Contains measurable variation; retain "
                    "for baseline modelling and later "
                    "feature-importance analysis."
                ),
            }
        )
    )

    selection_report = selection_report[
        [
            "sensor",
            "variance",
            "unique_values",
            "pearson_rul_correlation",
            "spearman_rul_correlation",
            "absolute_spearman_correlation",
            "selection_status",
            "selection_reason",
        ]
    ]

    selection_report = selection_report.sort_values(
        by=[
            "selection_status",
            "absolute_spearman_correlation",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return selection_report.reset_index(drop=True)


def plot_engine_lifespan_distribution(
    dataframe: pd.DataFrame,
) -> None:
    """
    Plot the distribution of engine lifespans.
    """

    engine_lifespans = (
        dataframe.groupby("engine_id")["cycle"]
        .max()
        .reset_index(name="lifespan")
    )

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=engine_lifespans,
        x="lifespan",
        bins=15,
        kde=True,
    )

    plt.title("FD001 Engine Lifespan Distribution")
    plt.xlabel("Total Operating Cycles")
    plt.ylabel("Number of Engines")
    plt.tight_layout()

    output_path = (
        EDA_OUTPUT_DIR
        / "engine_lifespan_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_sensor_variance(
    selection_report: pd.DataFrame,
) -> None:
    """
    Plot sensor variances using a logarithmic scale.
    """

    plot_data = selection_report.copy()

    plot_data = plot_data.sort_values(
        by="variance",
        ascending=True,
    )

    plt.figure(figsize=(11, 8))

    sns.barplot(
        data=plot_data,
        x="variance",
        y="sensor",
        hue="selection_status",
        dodge=False,
    )

    plt.xscale("symlog")

    plt.title(
        "FD001 Sensor Variance and Selection Status"
    )

    plt.xlabel("Variance - Symmetric Logarithmic Scale")
    plt.ylabel("Sensor")
    plt.legend(title="Decision")
    plt.tight_layout()

    output_path = (
        EDA_OUTPUT_DIR
        / "sensor_variance.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_sensor_rul_correlations(
    correlation_dataframe: pd.DataFrame,
) -> None:
    """
    Plot Spearman correlation between sensors and RUL.
    """

    plot_data = correlation_dataframe.sort_values(
        by="spearman_rul_correlation",
        ascending=True,
    )

    plt.figure(figsize=(11, 8))

    sns.barplot(
        data=plot_data,
        x="spearman_rul_correlation",
        y="sensor",
    )

    plt.axvline(
        x=0,
        color="black",
        linewidth=1,
    )

    plt.title(
        "Spearman Correlation Between Sensors and RUL"
    )

    plt.xlabel("Spearman Correlation")
    plt.ylabel("Sensor")
    plt.tight_layout()

    output_path = (
        EDA_OUTPUT_DIR
        / "sensor_rul_correlation.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_selected_sensor_heatmap(
    dataframe: pd.DataFrame,
    selected_sensors: list[str],
) -> None:
    """
    Plot the correlation matrix for retained sensors and RUL.
    """

    correlation_columns = (
        selected_sensors
        + ["rul"]
    )

    correlation_matrix = dataframe[
        correlation_columns
    ].corr(
        method="spearman"
    )

    plt.figure(figsize=(14, 11))

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        square=False,
        linewidths=0.3,
    )

    plt.title(
        "Selected Sensor Correlation Heatmap"
    )

    plt.tight_layout()

    output_path = (
        EDA_OUTPUT_DIR
        / "selected_sensor_correlation_heatmap.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_sensor_degradation_trends(
    dataframe: pd.DataFrame,
    selected_sensors: list[str],
) -> None:
    """
    Visualise degradation trends for selected engines.

    The plot uses the four highest-ranked retained sensors and
    three example engines with different lifespans.
    """

    correlation_dataframe = (
        calculate_sensor_rul_correlations(dataframe)
    )

    ranked_selected_sensors = (
        correlation_dataframe[
            correlation_dataframe["sensor"].isin(
                selected_sensors
            )
        ]
        .head(4)["sensor"]
        .tolist()
    )

    engine_lifespans = (
        dataframe.groupby("engine_id")["cycle"].max()
    )

    shortest_life_engine = int(
        engine_lifespans.idxmin()
    )

    longest_life_engine = int(
        engine_lifespans.idxmax()
    )

    median_lifespan = engine_lifespans.median()

    median_life_engine = int(
        (
            engine_lifespans - median_lifespan
        )
        .abs()
        .idxmin()
    )

    example_engines = [
        shortest_life_engine,
        median_life_engine,
        longest_life_engine,
    ]

    figure, axes = plt.subplots(
        nrows=len(ranked_selected_sensors),
        ncols=1,
        figsize=(13, 14),
        sharex=False,
    )

    if len(ranked_selected_sensors) == 1:
        axes = [axes]

    for axis, sensor_column in zip(
        axes,
        ranked_selected_sensors,
    ):
        for engine_id in example_engines:
            engine_data = dataframe[
                dataframe["engine_id"] == engine_id
            ]

            axis.plot(
                engine_data["cycle"],
                engine_data[sensor_column],
                label=f"Engine {engine_id}",
                linewidth=1.5,
            )

        axis.set_title(
            f"{sensor_column} Across Engine Lifecycles"
        )

        axis.set_xlabel("Operating Cycle")
        axis.set_ylabel("Sensor Reading")
        axis.legend()

    figure.suptitle(
        "Selected Sensor Degradation Trends",
        fontsize=15,
        y=1.01,
    )

    plt.tight_layout()

    output_path = (
        EDA_OUTPUT_DIR
        / "selected_sensor_degradation_trends.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_selected_sensor_list(
    selected_sensors: list[str],
) -> None:
    """
    Save selected sensor names for the modelling stage.
    """

    output_path = (
        EDA_OUTPUT_DIR
        / "selected_sensor_features.txt"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        for sensor in selected_sensors:
            output_file.write(f"{sensor}\n")


def run_eda_and_sensor_selection() -> None:
    """
    Run the complete FD001 exploratory analysis and
    sensor-selection pipeline.
    """

    prepare_output_directory()

    train_dataframe, _, _ = load_fd001_data()

    processed_train_dataframe = add_training_rul(
        train_dataframe
    )

    dataset_summary = create_dataset_summary(
        processed_train_dataframe
    )

    feature_statistics = calculate_feature_statistics(
        processed_train_dataframe
    )

    correlation_dataframe = (
        calculate_sensor_rul_correlations(
            processed_train_dataframe
        )
    )

    selection_report = create_sensor_selection_report(
        processed_train_dataframe,
        feature_statistics,
        correlation_dataframe,
    )

    selected_sensors = selection_report.loc[
        selection_report["selection_status"] == "Keep",
        "sensor",
    ].tolist()

    dropped_sensors = selection_report.loc[
        selection_report["selection_status"] == "Drop",
        "sensor",
    ].tolist()

    dataset_summary.to_csv(
        EDA_OUTPUT_DIR / "dataset_summary.csv",
        index=False,
    )

    feature_statistics.to_csv(
        EDA_OUTPUT_DIR / "feature_statistics.csv",
        index=False,
    )

    selection_report.to_csv(
        EDA_OUTPUT_DIR
        / "sensor_selection_report.csv",
        index=False,
    )

    save_selected_sensor_list(
        selected_sensors
    )

    plot_engine_lifespan_distribution(
        processed_train_dataframe
    )

    plot_sensor_variance(
        selection_report
    )

    plot_sensor_rul_correlations(
        correlation_dataframe
    )

    plot_selected_sensor_heatmap(
        processed_train_dataframe,
        selected_sensors,
    )

    plot_sensor_degradation_trends(
        processed_train_dataframe,
        selected_sensors,
    )

    print("\nFD001 EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\nDataset summary:")
    print(
        dataset_summary.to_string(
            index=False
        )
    )

    print("\nSelected sensors:")
    print(selected_sensors)

    print("\nDropped sensors:")
    print(dropped_sensors)

    print("\nSensor-selection ranking:")
    print(
        selection_report[
            [
                "sensor",
                "variance",
                "spearman_rul_correlation",
                "selection_status",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nEDA outputs saved to:"
    )

    print(EDA_OUTPUT_DIR)


if __name__ == "__main__":
    run_eda_and_sensor_selection()