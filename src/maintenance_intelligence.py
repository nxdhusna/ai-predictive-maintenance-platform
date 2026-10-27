from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "test_evaluation"
    / "official_test_predictions.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "maintenance_intelligence"
)

RUL_CAP = 125


def prepare_output_directory():
    """
    Create the maintenance-intelligence output directory.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_official_test_predictions():
    """
    Load predictions generated during official test evaluation.
    """

    if not TEST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Official test predictions were not found. "
            "Run 'python -m src.official_test_evaluation' first."
        )

    dataframe = pd.read_csv(
        TEST_PREDICTIONS_PATH
    )

    required_columns = {
        "engine_id",
        "final_observed_cycle",
        "predicted_rul",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def calculate_health_score(predicted_rul):
    """
    Convert predicted RUL into a health score from 0 to 100.

    A predicted RUL equal to or greater than the modelling cap
    receives a health score of 100. A predicted RUL of zero
    receives a health score of zero.
    """

    predicted_rul_array = np.asarray(
        predicted_rul,
        dtype=float,
    )

    health_score = (
        predicted_rul_array / RUL_CAP
    ) * 100

    health_score = np.clip(
        health_score,
        0,
        100,
    )

    return np.round(
        health_score,
        1,
    )


def calculate_risk_score(health_score):
    """
    Convert the health score into a maintenance-risk score.

    This is an interpretable operational risk index. It is not
    a statistically calibrated probability of failure.
    """

    health_score_array = np.asarray(
        health_score,
        dtype=float,
    )

    risk_score = 100 - health_score_array

    return np.round(
        np.clip(
            risk_score,
            0,
            100,
        ),
        1,
    )


def assign_health_status(health_score):
    """
    Assign an asset-health status using health-score thresholds.
    """

    if health_score >= 80:
        return "Healthy"

    if health_score >= 60:
        return "Monitor"

    if health_score >= 40:
        return "Maintenance Required"

    return "Critical"


def assign_risk_level(risk_score):
    """
    Assign a maintenance-risk category.
    """

    if risk_score >= 80:
        return "Critical"

    if risk_score >= 60:
        return "High"

    if risk_score >= 30:
        return "Moderate"

    return "Low"


def assign_maintenance_priority(predicted_rul):
    """
    Assign a maintenance priority based on predicted RUL.
    """

    if predicted_rul <= 10:
        return "P1 - Immediate"

    if predicted_rul <= 30:
        return "P2 - High"

    if predicted_rul <= 60:
        return "P3 - Scheduled"

    return "P4 - Routine"


def assign_maintenance_window(predicted_rul):
    """
    Recommend a maintenance-planning window.

    The recommendation is illustrative and intended only for
    this portfolio decision-support prototype.
    """

    if predicted_rul <= 10:
        return "Before the next 5 operating cycles"

    if predicted_rul <= 30:
        return "Within the next 10 operating cycles"

    if predicted_rul <= 60:
        return "Within the next 30 operating cycles"

    return "Continue routine monitoring"


def recommend_maintenance_action(predicted_rul):
    """
    Generate a transparent rule-based maintenance action.
    """

    if predicted_rul <= 10:
        return (
            "Conduct an immediate inspection and consider "
            "removing the asset from normal operation."
        )

    if predicted_rul <= 30:
        return (
            "Schedule high-priority maintenance and increase "
            "sensor-monitoring frequency."
        )

    if predicted_rul <= 60:
        return (
            "Plan preventive maintenance and review recent "
            "sensor-degradation trends."
        )

    return (
        "Continue normal operation with routine condition "
        "monitoring."
    )


def create_decision_explanation(
    predicted_rul,
    health_score,
    risk_level,
):
    """
    Create a concise explanation of the decision.
    """

    return (
        f"The model estimates {predicted_rul:.1f} remaining "
        f"operating cycles. This corresponds to an asset-health "
        f"score of {health_score:.1f}/100 and a {risk_level.lower()} "
        "maintenance-risk classification."
    )


def calculate_priority_rank(dataframe):
    """
    Rank assets from highest to lowest maintenance urgency.

    Lower predicted RUL and higher risk receive greater priority.
    """

    result = dataframe.copy()

    priority_mapping = {
        "P1 - Immediate": 1,
        "P2 - High": 2,
        "P3 - Scheduled": 3,
        "P4 - Routine": 4,
    }

    result["priority_number"] = (
        result["maintenance_priority"]
        .map(priority_mapping)
    )

    result = result.sort_values(
        by=[
            "priority_number",
            "predicted_rul",
            "risk_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    result["fleet_priority_rank"] = (
        range(1, len(result) + 1)
    )

    return result


def build_maintenance_intelligence(dataframe):
    """
    Build asset-health and maintenance-decision fields.
    """

    result = dataframe.copy()

    result["predicted_rul"] = (
        result["predicted_rul"]
        .clip(
            lower=0,
            upper=RUL_CAP,
        )
    )

    result["health_score"] = (
        calculate_health_score(
            result["predicted_rul"]
        )
    )

    result["risk_score"] = (
        calculate_risk_score(
            result["health_score"]
        )
    )

    result["health_status"] = (
        result["health_score"]
        .apply(assign_health_status)
    )

    result["risk_level"] = (
        result["risk_score"]
        .apply(assign_risk_level)
    )

    result["maintenance_priority"] = (
        result["predicted_rul"]
        .apply(assign_maintenance_priority)
    )

    result["maintenance_window"] = (
        result["predicted_rul"]
        .apply(assign_maintenance_window)
    )

    result["recommended_action"] = (
        result["predicted_rul"]
        .apply(recommend_maintenance_action)
    )

    result["decision_explanation"] = (
        result.apply(
            lambda row: create_decision_explanation(
                row["predicted_rul"],
                row["health_score"],
                row["risk_level"],
            ),
            axis=1,
        )
    )

    result = calculate_priority_rank(
        result
    )

    return result


def validate_maintenance_intelligence(dataframe):
    """
    Validate scoring outputs before saving them.
    """

    if dataframe.empty:
        raise ValueError(
            "Maintenance-intelligence data is empty."
        )

    if dataframe["health_score"].isna().any():
        raise ValueError(
            "Health scores contain missing values."
        )

    if dataframe["risk_score"].isna().any():
        raise ValueError(
            "Risk scores contain missing values."
        )

    if not dataframe["health_score"].between(
        0,
        100,
    ).all():
        raise ValueError(
            "Health scores must remain between 0 and 100."
        )

    if not dataframe["risk_score"].between(
        0,
        100,
    ).all():
        raise ValueError(
            "Risk scores must remain between 0 and 100."
        )

    if dataframe["engine_id"].duplicated().any():
        raise ValueError(
            "Each engine must appear only once."
        )

    expected_ranks = list(
        range(1, len(dataframe) + 1)
    )

    actual_ranks = (
        dataframe["fleet_priority_rank"]
        .tolist()
    )

    if actual_ranks != expected_ranks:
        raise ValueError(
            "Fleet-priority ranks are not sequential."
        )


def create_fleet_summary(dataframe):
    """
    Create high-level fleet-health KPIs.
    """

    health_counts = (
        dataframe["health_status"]
        .value_counts()
        .to_dict()
    )

    risk_counts = (
        dataframe["risk_level"]
        .value_counts()
        .to_dict()
    )

    priority_counts = (
        dataframe["maintenance_priority"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "total_assets": int(
            dataframe["engine_id"].nunique()
        ),
        "average_health_score": round(
            float(
                dataframe["health_score"].mean()
            ),
            2,
        ),
        "average_risk_score": round(
            float(
                dataframe["risk_score"].mean()
            ),
            2,
        ),
        "average_predicted_rul": round(
            float(
                dataframe["predicted_rul"].mean()
            ),
            2,
        ),
        "critical_assets": int(
            (
                dataframe["risk_level"]
                == "Critical"
            ).sum()
        ),
        "high_risk_assets": int(
            (
                dataframe["risk_level"]
                == "High"
            ).sum()
        ),
        "assets_requiring_action": int(
            dataframe[
                "maintenance_priority"
            ].isin(
                [
                    "P1 - Immediate",
                    "P2 - High",
                ]
            ).sum()
        ),
        "health_status_distribution": {
            key: int(value)
            for key, value in health_counts.items()
        },
        "risk_level_distribution": {
            key: int(value)
            for key, value in risk_counts.items()
        },
        "maintenance_priority_distribution": {
            key: int(value)
            for key, value in priority_counts.items()
        },
        "methodology_note": (
            "Risk scores are interpretable operational indices "
            "derived from predicted capped RUL. They are not "
            "calibrated probabilities of failure."
        ),
    }

    return summary


def plot_fleet_health_distribution(dataframe):
    """
    Plot the number of assets in each health category.
    """

    category_order = [
        "Healthy",
        "Monitor",
        "Maintenance Required",
        "Critical",
    ]

    plot_data = (
        dataframe["health_status"]
        .value_counts()
        .reindex(
            category_order,
            fill_value=0,
        )
        .reset_index()
    )

    plot_data.columns = [
        "health_status",
        "asset_count",
    ]

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=plot_data,
        x="health_status",
        y="asset_count",
        order=category_order,
    )

    plt.title(
        "Official FD001 Fleet Health Distribution"
    )

    plt.xlabel("Asset Health Status")
    plt.ylabel("Number of Assets")
    plt.xticks(
        rotation=15,
        ha="right",
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "fleet_health_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_priority_queue(
    dataframe,
    number_of_assets=20,
):
    """
    Visualise the highest-priority assets.
    """

    plot_data = (
        dataframe.head(number_of_assets)
        .sort_values(
            "risk_score",
            ascending=True,
        )
        .copy()
    )

    plot_data["engine_label"] = (
        "Engine "
        + plot_data["engine_id"]
        .astype(int)
        .astype(str)
    )

    plt.figure(figsize=(11, 8))

    sns.barplot(
        data=plot_data,
        x="risk_score",
        y="engine_label",
        hue="risk_level",
        dodge=False,
    )

    plt.title(
        "Top Maintenance-Priority Assets"
    )

    plt.xlabel(
        "Maintenance Risk Score"
    )

    plt.ylabel("Asset")
    plt.xlim(0, 100)
    plt.legend(title="Risk Level")
    plt.tight_layout()

    output_path = (
        OUTPUT_DIRECTORY
        / "maintenance_priority_queue.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_maintenance_outputs(
    dataframe,
    fleet_summary,
):
    """
    Save detailed scoring, priority queue, and fleet summary.
    """

    detailed_output_path = (
        OUTPUT_DIRECTORY
        / "official_test_asset_health.csv"
    )

    priority_output_path = (
        OUTPUT_DIRECTORY
        / "maintenance_priority_queue.csv"
    )

    summary_output_path = (
        OUTPUT_DIRECTORY
        / "fleet_health_summary.json"
    )

    dataframe.to_csv(
        detailed_output_path,
        index=False,
    )

    priority_columns = [
        "fleet_priority_rank",
        "engine_id",
        "final_observed_cycle",
        "predicted_rul",
        "health_score",
        "health_status",
        "risk_score",
        "risk_level",
        "maintenance_priority",
        "maintenance_window",
        "recommended_action",
    ]

    dataframe[
        priority_columns
    ].to_csv(
        priority_output_path,
        index=False,
    )

    with summary_output_path.open(
        mode="w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(
            fleet_summary,
            summary_file,
            indent=4,
        )


def print_top_priority_assets(
    dataframe,
    number_of_assets=10,
):
    """
    Print the highest-priority assets to the terminal.
    """

    display_columns = [
        "fleet_priority_rank",
        "engine_id",
        "predicted_rul",
        "health_score",
        "risk_score",
        "risk_level",
        "maintenance_priority",
    ]

    print(
        dataframe[
            display_columns
        ]
        .head(number_of_assets)
        .to_string(index=False)
    )


def run_maintenance_intelligence():
    """
    Run the complete asset-health and maintenance-risk pipeline.
    """

    prepare_output_directory()

    prediction_dataframe = (
        load_official_test_predictions()
    )

    maintenance_dataframe = (
        build_maintenance_intelligence(
            prediction_dataframe
        )
    )

    validate_maintenance_intelligence(
        maintenance_dataframe
    )

    fleet_summary = create_fleet_summary(
        maintenance_dataframe
    )

    save_maintenance_outputs(
        maintenance_dataframe,
        fleet_summary,
    )

    plot_fleet_health_distribution(
        maintenance_dataframe
    )

    plot_priority_queue(
        maintenance_dataframe
    )

    print("\nASSET HEALTH AND MAINTENANCE INTELLIGENCE")
    print("=" * 75)

    print(
        f"Total assets: "
        f"{fleet_summary['total_assets']}"
    )

    print(
        f"Average health score: "
        f"{fleet_summary['average_health_score']}/100"
    )

    print(
        f"Average risk score: "
        f"{fleet_summary['average_risk_score']}%"
    )

    print(
        f"Average predicted RUL: "
        f"{fleet_summary['average_predicted_rul']} cycles"
    )

    print(
        f"Critical assets: "
        f"{fleet_summary['critical_assets']}"
    )

    print(
        f"High-risk assets: "
        f"{fleet_summary['high_risk_assets']}"
    )

    print(
        f"Assets requiring immediate or "
        f"high-priority action: "
        f"{fleet_summary['assets_requiring_action']}"
    )

    print("\nTOP MAINTENANCE PRIORITIES")
    print("-" * 75)

    print_top_priority_assets(
        maintenance_dataframe
    )

    print("\nImportant interpretation:")
    print(
        "The risk score is an operational risk index derived "
        "from predicted RUL. It is not a calibrated probability "
        "of failure."
    )

    print("\nOutputs saved to:")
    print(OUTPUT_DIRECTORY)


if __name__ == "__main__":
    run_maintenance_intelligence()