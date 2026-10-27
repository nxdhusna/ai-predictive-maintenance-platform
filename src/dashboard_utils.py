from pathlib import Path
import json

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAINTENANCE_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "maintenance_intelligence"
)

ANOMALY_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "anomaly_detection"
)

MODEL_EVALUATION_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "model_evaluation"
)

TEST_EVALUATION_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "test_evaluation"
)

FAILURE_CLASSIFICATION_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "failure_classification"
)

EXPLAINABILITY_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "explainability"
)



@st.cache_data
def load_csv_file(file_path):
    """
    Load and cache a CSV file used by the dashboard.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required dashboard file was not found: {path}"
        )

    return pd.read_csv(path)


@st.cache_data
def load_json_file(file_path):
    """
    Load and cache a JSON file used by the dashboard.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required dashboard file was not found: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as json_file:
        return json.load(json_file)


def load_asset_health_data():
    return load_csv_file(
        MAINTENANCE_DIRECTORY
        / "official_test_asset_health.csv"
    )


def load_priority_queue():
    return load_csv_file(
        MAINTENANCE_DIRECTORY
        / "maintenance_priority_queue.csv"
    )


def load_fleet_summary():
    return load_json_file(
        MAINTENANCE_DIRECTORY
        / "fleet_health_summary.json"
    )


def load_anomaly_results():
    return load_csv_file(
        ANOMALY_DIRECTORY
        / "final_cycle_anomaly_results.csv"
    )


def load_model_comparison():
    return load_csv_file(
        MODEL_EVALUATION_DIRECTORY
        / "model_comparison.csv"
    )


def load_terminal_cycle_evaluation():
    return load_csv_file(
        MODEL_EVALUATION_DIRECTORY
        / "terminal_cycle_evaluation.csv"
    )


def load_official_test_predictions():
    return load_csv_file(
        TEST_EVALUATION_DIRECTORY
        / "official_test_predictions.csv"
    )


def load_failure_metrics():
    return load_json_file(
        FAILURE_CLASSIFICATION_DIRECTORY
        / "classification_metrics.json"
    )


def combine_asset_and_anomaly_data():
    """
    Combine health, maintenance, anomaly, and failure-
    probability fields by engine.
    """

    asset_health = load_asset_health_data()
    anomaly_results = load_anomaly_results()

    anomaly_columns = [
        "engine_id",
        "anomaly_score",
        "anomaly_status",
        "anomaly_severity",
        "anomaly_decision_score",
        "anomaly_explanation",
    ]

    available_anomaly_columns = [
        column
        for column in anomaly_columns
        if column in anomaly_results.columns
    ]

    combined_dataframe = asset_health.merge(
        anomaly_results[
            available_anomaly_columns
        ],
        on="engine_id",
        how="left",
        validate="one_to_one",
    )

    failure_probability_path = (
        FAILURE_CLASSIFICATION_DIRECTORY
        / "official_test_failure_probabilities.csv"
    )

    if failure_probability_path.exists():
        failure_probabilities = (
            load_official_failure_probabilities()
        )

        failure_columns = [
            "engine_id",
            "failure_probability",
            "failure_probability_percent",
            "predicted_within_30_cycles",
        ]

        available_failure_columns = [
            column
            for column in failure_columns
            if column
            in failure_probabilities.columns
        ]

        combined_dataframe = (
            combined_dataframe.merge(
                failure_probabilities[
                    available_failure_columns
                ],
                on="engine_id",
                how="left",
                validate="one_to_one",
            )
        )

    return combined_dataframe

def format_percentage(value):
    """
    Format a numeric value as a percentage.
    """

    return f"{float(value):.1f}%"


def format_cycles(value):
    """
    Format an RUL value using operating-cycle units.
    """

    return f"{float(value):.1f} cycles"


def render_project_disclaimer():
    """
    Show the portfolio and safety disclaimer.
    """

    st.warning(
        "Portfolio demonstration only. This platform uses "
        "simulated NASA C-MAPSS benchmark data and is not "
        "validated for real-world aviation, automotive, or "
        "safety-critical maintenance decisions."
    )


def assign_status_colour(status):
    """
    Return a display colour for a risk or health status.
    """

    colour_mapping = {
        "Healthy": "#2E8B57",
        "Monitor": "#D4A017",
        "Maintenance Required": "#E67E22",
        "Critical": "#C0392B",
        "Low": "#2E8B57",
        "Moderate": "#D4A017",
        "High": "#E67E22",
        "Normal": "#2E8B57",
    }

    return colour_mapping.get(
        status,
        "#5D6D7E",
    )


def render_sidebar_information():
    """
    Display common project information in the sidebar.
    """

    with st.sidebar:
        st.markdown("### Platform Information")

        st.markdown(
            """
            **Dataset:** NASA C-MAPSS FD001  
            **Assets:** 100 simulated turbofan engines  
            **Primary model:** Random Forest  
            **RUL cap:** 125 cycles  
            **Failure horizon:** 30 cycles
            """
        )

        st.divider()

        st.caption(
            "Developed as an AI predictive-maintenance "
            "portfolio project."
        )

def load_global_feature_importance():
    return load_csv_file(
        EXPLAINABILITY_DIRECTORY
        / "global_feature_importance.csv"
    )


def load_local_asset_explanations():
    return load_csv_file(
        EXPLAINABILITY_DIRECTORY
        / "local_asset_explanations.csv"
    )


def load_local_explanation_summary():
    return load_csv_file(
        EXPLAINABILITY_DIRECTORY
        / "local_explanation_summary.csv"
    )


def load_explainability_metadata():
    return load_json_file(
        EXPLAINABILITY_DIRECTORY
        / "explainability_metadata.json"
    )

def load_official_failure_probabilities():
    return load_csv_file(
        FAILURE_CLASSIFICATION_DIRECTORY
        / "official_test_failure_probabilities.csv"
    )
