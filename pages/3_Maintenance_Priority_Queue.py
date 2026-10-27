import plotly.express as px
import streamlit as st

from src.dashboard_utils import (
    combine_asset_and_anomaly_data,
    render_project_disclaimer,
    render_sidebar_information,
)


st.set_page_config(
    page_title="Maintenance Priority Queue",
    page_icon="🛠️",
    layout="wide",
)


render_sidebar_information()

st.title("Maintenance Priority Queue")

st.markdown(
    """
    Rank assets by maintenance urgency using predicted RUL,
    health score, risk classification, and anomaly information.
    """
)

render_project_disclaimer()


try:
    priority_dataframe = (
        combine_asset_and_anomaly_data()
    )

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


priority_options = [
    "P1 - Immediate",
    "P2 - High",
    "P3 - Scheduled",
    "P4 - Routine",
]

selected_priorities = st.multiselect(
    label="Maintenance Priority",
    options=priority_options,
    default=priority_options,
)


filtered_dataframe = priority_dataframe[
    priority_dataframe[
        "maintenance_priority"
    ].isin(selected_priorities)
].copy()


if filtered_dataframe.empty:
    st.info(
        "No assets match the selected priorities."
    )

    st.stop()


metric_1, metric_2, metric_3 = st.columns(3)

with metric_1:
    p1_count = int(
        (
            filtered_dataframe[
                "maintenance_priority"
            ]
            == "P1 - Immediate"
        ).sum()
    )

    st.metric(
        "Immediate Actions",
        p1_count,
    )

with metric_2:
    p2_count = int(
        (
            filtered_dataframe[
                "maintenance_priority"
            ]
            == "P2 - High"
        ).sum()
    )

    st.metric(
        "High-Priority Actions",
        p2_count,
    )

with metric_3:
    anomalous_count = int(
        (
            filtered_dataframe[
                "anomaly_status"
            ]
            == "Anomaly Detected"
        ).sum()
    )

    st.metric(
        "Assets with Anomalies",
        anomalous_count,
    )


chart_dataframe = (
    filtered_dataframe
    .sort_values(
        "fleet_priority_rank"
    )
    .head(25)
    .copy()
)

chart_dataframe["Engine"] = (
    "Engine "
    + chart_dataframe[
        "engine_id"
    ].astype(int).astype(str)
)

priority_chart = px.bar(
    chart_dataframe,
    x="risk_score",
    y="Engine",
    color="maintenance_priority",
    orientation="h",
    hover_data=[
        "predicted_rul",
        "health_score",
        "anomaly_score",
    ],
    title="Top Maintenance-Priority Assets",
    labels={
        "risk_score": "Maintenance Risk Score",
        "maintenance_priority": (
            "Maintenance Priority"
        ),
    },
)

priority_chart.update_layout(
    yaxis={
        "categoryorder": "total ascending"
    }
)

st.plotly_chart(
    priority_chart,
    use_container_width=True,
)


st.subheader("Action Queue")

display_columns = [
    "fleet_priority_rank",
    "engine_id",
    "predicted_rul",
    "health_score",
    "risk_score",
    "risk_level",
    "maintenance_priority",
    "maintenance_window",
    "anomaly_score",
    "anomaly_status",
    "recommended_action",
]

available_columns = [
    column
    for column in display_columns
    if column in filtered_dataframe.columns
]

queue_dataframe = (
    filtered_dataframe[
        available_columns
    ]
    .sort_values(
        "fleet_priority_rank"
    )
    .copy()
)

queue_dataframe = queue_dataframe.rename(
    columns={
        "fleet_priority_rank": "Priority Rank",
        "engine_id": "Engine ID",
        "predicted_rul": "Predicted RUL",
        "health_score": "Health Score",
        "risk_score": "Risk Score",
        "risk_level": "Risk Level",
        "maintenance_priority": (
            "Maintenance Priority"
        ),
        "maintenance_window": (
            "Maintenance Window"
        ),
        "anomaly_score": "Anomaly Score",
        "anomaly_status": "Anomaly Status",
        "recommended_action": (
            "Recommended Action"
        ),
    }
)

st.dataframe(
    queue_dataframe,
    use_container_width=True,
    hide_index=True,
)


csv_data = queue_dataframe.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Maintenance Priority Queue",
    data=csv_data,
    file_name="maintenance_priority_queue.csv",
    mime="text/csv",
)