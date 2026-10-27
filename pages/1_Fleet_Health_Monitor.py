import plotly.express as px
import streamlit as st

from src.dashboard_utils import (
    combine_asset_and_anomaly_data,
    render_project_disclaimer,
    render_sidebar_information,
)


st.set_page_config(
    page_title="Fleet Health Monitor",
    page_icon="📊",
    layout="wide",
)


render_sidebar_information()

st.title("Fleet Health Monitor")

st.markdown(
    """
    Explore the health, maintenance risk, predicted Remaining
    Useful Life, and anomaly status of the complete test fleet.
    """
)

render_project_disclaimer()


try:
    fleet_dataframe = (
        combine_asset_and_anomaly_data()
    )

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


filter_column_1, filter_column_2, filter_column_3 = (
    st.columns(3)
)

with filter_column_1:
    selected_health_status = st.multiselect(
        label="Health Status",
        options=sorted(
            fleet_dataframe[
                "health_status"
            ].dropna().unique()
        ),
        default=sorted(
            fleet_dataframe[
                "health_status"
            ].dropna().unique()
        ),
    )

with filter_column_2:
    selected_risk_levels = st.multiselect(
        label="Risk Level",
        options=sorted(
            fleet_dataframe[
                "risk_level"
            ].dropna().unique()
        ),
        default=sorted(
            fleet_dataframe[
                "risk_level"
            ].dropna().unique()
        ),
    )

with filter_column_3:
    selected_anomaly_status = st.multiselect(
        label="Anomaly Status",
        options=sorted(
            fleet_dataframe[
                "anomaly_status"
            ].dropna().unique()
        ),
        default=sorted(
            fleet_dataframe[
                "anomaly_status"
            ].dropna().unique()
        ),
    )


filtered_dataframe = fleet_dataframe[
    fleet_dataframe["health_status"].isin(
        selected_health_status
    )
    & fleet_dataframe["risk_level"].isin(
        selected_risk_levels
    )
    & fleet_dataframe["anomaly_status"].isin(
        selected_anomaly_status
    )
].copy()


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Displayed Assets",
        len(filtered_dataframe),
    )

with metric_2:
    average_health = (
        filtered_dataframe["health_score"].mean()
        if not filtered_dataframe.empty
        else 0
    )

    st.metric(
        "Average Health",
        f"{average_health:.1f}/100",
    )

with metric_3:
    average_risk = (
        filtered_dataframe["risk_score"].mean()
        if not filtered_dataframe.empty
        else 0
    )

    st.metric(
        "Average Risk",
        f"{average_risk:.1f}%",
    )

with metric_4:
    anomaly_count = int(
        (
            filtered_dataframe["anomaly_status"]
            == "Anomaly Detected"
        ).sum()
    )

    st.metric(
        "Anomalous Assets",
        anomaly_count,
    )


st.divider()

if filtered_dataframe.empty:
    st.info(
        "No assets match the selected filters."
    )

    st.stop()


scatter_chart = px.scatter(
    filtered_dataframe,
    x="predicted_rul",
    y="health_score",
    color="risk_level",
    size="anomaly_score",
    hover_data=[
        "engine_id",
        "maintenance_priority",
        "anomaly_status",
        "anomaly_severity",
    ],
    color_discrete_map={
        "Low": "#2E8B57",
        "Moderate": "#D4A017",
        "High": "#E67E22",
        "Critical": "#C0392B",
    },
    title=(
        "Predicted Remaining Useful Life "
        "vs Asset Health"
    ),
    labels={
        "predicted_rul": "Predicted RUL",
        "health_score": "Health Score",
        "risk_level": "Risk Level",
        "anomaly_score": "Anomaly Score",
    },
)

st.plotly_chart(
    scatter_chart,
    use_container_width=True,
)


st.subheader("Fleet Asset Table")

display_columns = [
    "fleet_priority_rank",
    "engine_id",
    "predicted_rul",
    "health_score",
    "health_status",
    "risk_score",
    "risk_level",
    "maintenance_priority",
    "anomaly_score",
    "anomaly_status",
]

available_columns = [
    column
    for column in display_columns
    if column in filtered_dataframe.columns
]

display_dataframe = (
    filtered_dataframe[
        available_columns
    ]
    .sort_values(
        "fleet_priority_rank"
    )
    .copy()
)

display_dataframe = display_dataframe.rename(
    columns={
        "fleet_priority_rank": "Priority Rank",
        "engine_id": "Engine ID",
        "predicted_rul": "Predicted RUL",
        "health_score": "Health Score",
        "health_status": "Health Status",
        "risk_score": "Risk Score",
        "risk_level": "Risk Level",
        "maintenance_priority": (
            "Maintenance Priority"
        ),
        "anomaly_score": "Anomaly Score",
        "anomaly_status": "Anomaly Status",
    }
)

st.dataframe(
    display_dataframe,
    use_container_width=True,
    hide_index=True,
)


csv_data = display_dataframe.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Filtered Fleet Data",
    data=csv_data,
    file_name="filtered_fleet_health.csv",
    mime="text/csv",
)