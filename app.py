import plotly.express as px
import streamlit as st

from src.dashboard_utils import (
    combine_asset_and_anomaly_data,
    load_fleet_summary,
    render_project_disclaimer,
    render_sidebar_information,
)


st.set_page_config(
    page_title="Predictive Maintenance Intelligence",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


render_sidebar_information()

st.title(
    "AI-Driven Asset Health and Predictive "
    "Maintenance Platform"
)

st.markdown(
    """
    An explainable AI decision-support platform for monitoring
    equipment health, estimating Remaining Useful Life, detecting
    unusual sensor behaviour, and prioritising maintenance actions.
    """
)

render_project_disclaimer()


try:
    fleet_summary = load_fleet_summary()
    asset_dataframe = combine_asset_and_anomaly_data()

except FileNotFoundError as error:
    st.error(str(error))

    st.info(
        "Run the modelling and maintenance-intelligence "
        "pipelines before launching the dashboard."
    )

    st.stop()


st.subheader("Executive Fleet Overview")

column_1, column_2, column_3, column_4 = st.columns(4)

with column_1:
    st.metric(
        label="Assets Monitored",
        value=fleet_summary["total_assets"],
    )

with column_2:
    st.metric(
        label="Average Health Score",
        value=(
            f"{fleet_summary['average_health_score']}/100"
        ),
    )

with column_3:
    st.metric(
        label="Critical Assets",
        value=fleet_summary["critical_assets"],
    )

with column_4:
    st.metric(
        label="Average Predicted RUL",
        value=(
            f"{fleet_summary['average_predicted_rul']} cycles"
        ),
    )


st.divider()

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Fleet Health Distribution")

    health_order = [
        "Healthy",
        "Monitor",
        "Maintenance Required",
        "Critical",
    ]

    health_distribution = (
        asset_dataframe["health_status"]
        .value_counts()
        .reindex(
            health_order,
            fill_value=0,
        )
        .reset_index()
    )

    health_distribution.columns = [
        "Health Status",
        "Asset Count",
    ]

    health_chart = px.bar(
        health_distribution,
        x="Health Status",
        y="Asset Count",
        color="Health Status",
        color_discrete_map={
            "Healthy": "#2E8B57",
            "Monitor": "#D4A017",
            "Maintenance Required": "#E67E22",
            "Critical": "#C0392B",
        },
        title="Assets by Health Status",
    )

    health_chart.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        health_chart,
        use_container_width=True,
    )


with right_column:
    st.subheader("Maintenance Risk Distribution")

    risk_order = [
        "Low",
        "Moderate",
        "High",
        "Critical",
    ]

    risk_distribution = (
        asset_dataframe["risk_level"]
        .value_counts()
        .reindex(
            risk_order,
            fill_value=0,
        )
        .reset_index()
    )

    risk_distribution.columns = [
        "Risk Level",
        "Asset Count",
    ]

    risk_chart = px.bar(
        risk_distribution,
        x="Risk Level",
        y="Asset Count",
        color="Risk Level",
        color_discrete_map={
            "Low": "#2E8B57",
            "Moderate": "#D4A017",
            "High": "#E67E22",
            "Critical": "#C0392B",
        },
        title="Assets by Maintenance Risk",
    )

    risk_chart.update_layout(
        showlegend=False
    )

    st.plotly_chart(
        risk_chart,
        use_container_width=True,
    )


st.divider()

st.subheader("Highest-Priority Assets")

priority_columns = [
    "fleet_priority_rank",
    "engine_id",
    "predicted_rul",
    "health_score",
    "risk_score",
    "risk_level",
    "maintenance_priority",
]

available_priority_columns = [
    column
    for column in priority_columns
    if column in asset_dataframe.columns
]

top_priority_assets = (
    asset_dataframe
    .sort_values(
        "fleet_priority_rank"
    )
    .head(10)[available_priority_columns]
    .copy()
)

top_priority_assets = top_priority_assets.rename(
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
    }
)

st.dataframe(
    top_priority_assets,
    use_container_width=True,
    hide_index=True,
)


st.divider()

st.subheader("Platform Intelligence")

intelligence_1, intelligence_2, intelligence_3 = (
    st.columns(3)
)

with intelligence_1:
    st.markdown("#### RUL Prediction")

    st.write(
        "Estimates the remaining operating cycles for each "
        "asset using a Random Forest regression model."
    )

with intelligence_2:
    st.markdown("#### Failure Classification")

    st.write(
        "Estimates whether an asset has entered the final "
        "30-cycle simulated failure horizon."
    )

with intelligence_3:
    st.markdown("#### Anomaly Detection")

    st.write(
        "Uses Isolation Forest to identify unusual sensor "
        "patterns relative to healthy reference behaviour."
    )