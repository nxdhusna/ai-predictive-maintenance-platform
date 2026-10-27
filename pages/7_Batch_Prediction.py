from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.batch_prediction import (
    create_batch_predictions,
    create_batch_summary,
    load_prediction_models,
    validate_uploaded_data,
)
from src.dashboard_utils import (
    render_project_disclaimer,
    render_sidebar_information,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SAMPLE_FILE_PATH = (
    PROJECT_ROOT
    / "data"
    / "samples"
    / "sample_batch_sensor_history.csv"
)


st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📤",
    layout="wide",
)


render_sidebar_information()

st.title("Batch Sensor Prediction")

st.markdown(
    """
    Upload chronological sensor histories for one or more assets.
    The platform will validate the data, create time-aware
    features, and generate asset-level maintenance intelligence.
    """
)

render_project_disclaimer()


st.info(
    "Each asset should contain multiple chronological sensor "
    "observations. The platform generates one prediction using "
    "the latest uploaded cycle for each asset."
)


download_column, upload_column = st.columns(2)


with download_column:
    st.subheader("1. Download Sample Data")

    st.write(
        "Use the sample file to test the complete workflow and "
        "understand the required column structure."
    )

    if SAMPLE_FILE_PATH.exists():
        sample_bytes = (
            SAMPLE_FILE_PATH.read_bytes()
        )

        st.download_button(
            label="Download Sample Sensor History",
            data=sample_bytes,
            file_name=(
                "sample_batch_sensor_history.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    else:
        st.warning(
            "Sample upload file is unavailable. Run:"
        )

        st.code(
            "python -m src.create_batch_sample"
        )


with upload_column:
    st.subheader("2. Upload Sensor Data")

    uploaded_file = st.file_uploader(
        label="Upload a CSV sensor-history file",
        type=["csv"],
        accept_multiple_files=False,
        help=(
            "The CSV must contain engine_id, cycle, "
            "three operational settings, and sensor_1 "
            "through sensor_21."
        ),
    )


if uploaded_file is None:
    st.subheader("Required Columns")

    required_columns = [
        "engine_id",
        "cycle",
        "operational_setting_1",
        "operational_setting_2",
        "operational_setting_3",
        "sensor_1 through sensor_21",
    ]

    for column in required_columns:
        st.write(f"- `{column}`")

    st.stop()


try:
    uploaded_dataframe = pd.read_csv(
        uploaded_file
    )

except Exception as error:
    st.error(
        "The uploaded file could not be read as CSV."
    )

    st.exception(error)
    st.stop()


st.subheader("3. Validate Uploaded Data")


try:
    (
        validated_dataframe,
        validation_warnings,
        validation_summary,
    ) = validate_uploaded_data(
        uploaded_dataframe
    )

except ValueError as error:
    st.error(
        f"Data validation failed: {error}"
    )

    st.stop()


st.success(
    "The uploaded file passed all required "
    "schema and data-quality checks."
)


summary_1, summary_2, summary_3, summary_4 = (
    st.columns(4)
)

with summary_1:
    st.metric(
        "Uploaded Rows",
        validation_summary["rows"],
    )

with summary_2:
    st.metric(
        "Assets Found",
        validation_summary["assets"],
    )

with summary_3:
    st.metric(
        "Minimum Cycle",
        validation_summary[
            "minimum_cycle"
        ],
    )

with summary_4:
    st.metric(
        "Maximum Cycle",
        validation_summary[
            "maximum_cycle"
        ],
    )


for warning_message in validation_warnings:
    st.warning(warning_message)


with st.expander(
    "Preview validated sensor data"
):
    st.dataframe(
        validated_dataframe.head(100),
        use_container_width=True,
        hide_index=True,
    )


st.subheader("4. Generate Predictions")


if st.button(
    "Run AI Maintenance Analysis",
    type="primary",
    use_container_width=True,
):
    try:
        with st.spinner(
            "Generating features and scoring assets..."
        ):
            model_bundle = (
                load_prediction_models()
            )

            (
                prediction_dataframe,
                feature_dataframe,
            ) = create_batch_predictions(
                validated_dataframe,
                model_bundle,
            )

            prediction_summary = (
                create_batch_summary(
                    prediction_dataframe
                )
            )

            st.session_state[
                "batch_predictions"
            ] = prediction_dataframe

            st.session_state[
                "batch_feature_rows"
            ] = len(feature_dataframe)

            st.session_state[
                "batch_summary"
            ] = prediction_summary

    except Exception as error:
        st.error(
            "Batch scoring could not be completed."
        )

        st.exception(error)


if (
    "batch_predictions"
    not in st.session_state
):
    st.stop()


prediction_dataframe = st.session_state[
    "batch_predictions"
]

prediction_summary = st.session_state[
    "batch_summary"
]


st.success(
    "Batch scoring completed successfully."
)


result_1, result_2, result_3, result_4 = (
    st.columns(4)
)

with result_1:
    st.metric(
        "Assets Scored",
        prediction_summary[
            "assets_scored"
        ],
    )

with result_2:
    st.metric(
        "Average Predicted RUL",
        (
            f"{prediction_summary[
                'average_predicted_rul'
            ]:.1f} cycles"
        ),
    )

with result_3:
    st.metric(
        "Immediate Priorities",
        prediction_summary[
            "p1_assets"
        ],
    )

with result_4:
    st.metric(
        "Anomalous Assets",
        prediction_summary[
            "anomalous_assets"
        ],
    )


st.divider()

st.subheader("Batch Maintenance Intelligence")


chart_dataframe = (
    prediction_dataframe.copy()
)

chart_dataframe["Asset"] = (
    "Engine "
    + chart_dataframe[
        "engine_id"
    ].astype(int).astype(str)
)


prediction_chart = px.scatter(
    chart_dataframe,
    x="predicted_rul",
    y="failure_probability_percent",
    color="risk_level",
    size="anomaly_score",
    hover_data=[
        "engine_id",
        "maintenance_priority",
        "health_status",
        "anomaly_status",
    ],
    color_discrete_map={
        "Low": "#2E8B57",
        "Moderate": "#D4A017",
        "High": "#E67E22",
        "Critical": "#C0392B",
    },
    title=(
        "Predicted RUL vs Failure-Horizon Probability"
    ),
    labels={
        "predicted_rul": (
            "Predicted Remaining Useful Life"
        ),
        "failure_probability_percent": (
            "Failure Within 30 Cycles (%)"
        ),
        "risk_level": "Risk Level",
        "anomaly_score": "Anomaly Score",
    },
)


st.plotly_chart(
    prediction_chart,
    use_container_width=True,
)


display_columns = [
    "batch_priority_rank",
    "engine_id",
    "latest_observed_cycle",
    "predicted_rul",
    "failure_probability_percent",
    "health_score",
    "health_status",
    "risk_score",
    "risk_level",
    "anomaly_score",
    "anomaly_status",
    "anomaly_severity",
    "maintenance_priority",
    "maintenance_window",
    "recommended_action",
    "history_rows_used",
]


available_columns = [
    column
    for column in display_columns
    if column
    in prediction_dataframe.columns
]


display_dataframe = (
    prediction_dataframe[
        available_columns
    ]
    .copy()
)


display_dataframe = (
    display_dataframe.rename(
        columns={
            "batch_priority_rank": (
                "Priority Rank"
            ),
            "engine_id": "Engine ID",
            "latest_observed_cycle": (
                "Latest Cycle"
            ),
            "predicted_rul": (
                "Predicted RUL"
            ),
            "failure_probability_percent": (
                "Failure Within 30 Cycles (%)"
            ),
            "health_score": (
                "Health Score"
            ),
            "health_status": (
                "Health Status"
            ),
            "risk_score": (
                "Risk Score"
            ),
            "risk_level": (
                "Risk Level"
            ),
            "anomaly_score": (
                "Anomaly Score"
            ),
            "anomaly_status": (
                "Anomaly Status"
            ),
            "anomaly_severity": (
                "Anomaly Severity"
            ),
            "maintenance_priority": (
                "Maintenance Priority"
            ),
            "maintenance_window": (
                "Maintenance Window"
            ),
            "recommended_action": (
                "Recommended Action"
            ),
            "history_rows_used": (
                "History Rows Used"
            ),
        }
    )
)


st.dataframe(
    display_dataframe,
    use_container_width=True,
    hide_index=True,
)


download_data = (
    display_dataframe
    .to_csv(
        index=False
    )
    .encode("utf-8")
)


st.download_button(
    label="Download Enriched Prediction Results",
    data=download_data,
    file_name=(
        "batch_predictive_maintenance_results.csv"
    ),
    mime="text/csv",
    type="primary",
    use_container_width=True,
)


st.divider()

st.subheader("Interpretation Notes")

st.markdown(
    """
    - **Predicted RUL** estimates remaining operating cycles up to
      the model cap of 125 cycles.
    - **Failure Within 30 Cycles** is the calibrated classifier
      estimate for the final 30-cycle simulated horizon.
    - **Risk Score** is an operational index derived from
      predicted RUL, not a probability.
    - **Anomaly Score** measures how unusual the latest sensor
      pattern is relative to the healthy training reference.
    - **Maintenance Priority** is generated from transparent
      decision rules.
    """
)