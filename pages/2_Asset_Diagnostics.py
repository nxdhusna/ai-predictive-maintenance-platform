import plotly.graph_objects as go
import streamlit as st

from src.dashboard_utils import (
    combine_asset_and_anomaly_data,
    render_project_disclaimer,
    render_sidebar_information,
)


st.set_page_config(
    page_title="Asset Diagnostics",
    page_icon="🔍",
    layout="wide",
)


render_sidebar_information()

st.title("Asset Diagnostics")

st.markdown(
    """
    Examine one asset's predicted Remaining Useful Life,
    health score, risk classification, anomaly status,
    and recommended maintenance action.
    """
)

render_project_disclaimer()


try:
    asset_dataframe = (
        combine_asset_and_anomaly_data()
    )

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


engine_ids = sorted(
    asset_dataframe["engine_id"]
    .astype(int)
    .unique()
)

selected_engine_id = st.selectbox(
    label="Select Engine",
    options=engine_ids,
    format_func=lambda engine_id: (
        f"Engine {engine_id}"
    ),
)


selected_asset = asset_dataframe.loc[
    asset_dataframe["engine_id"]
    == selected_engine_id
].iloc[0]


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Predicted RUL",
        (
            f"{selected_asset['predicted_rul']:.1f} "
            "cycles"
        ),
    )

with metric_2:
    st.metric(
        "Health Score",
        (
            f"{selected_asset['health_score']:.1f}"
            "/100"
        ),
    )

with metric_3:
    st.metric(
        "Maintenance Risk",
        (
            f"{selected_asset['risk_score']:.1f}%"
        ),
    )

with metric_4:
    anomaly_score = selected_asset.get(
        "anomaly_score",
        0,
    )

    st.metric(
        "Anomaly Score",
        f"{anomaly_score:.1f}/100",
    )


st.divider()

gauge_column_1, gauge_column_2 = st.columns(2)

with gauge_column_1:
    health_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=selected_asset[
                "health_score"
            ],
            title={
                "text": "Asset Health Score"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "steps": [
                    {
                        "range": [0, 40],
                        "color": "#F5B7B1",
                    },
                    {
                        "range": [40, 60],
                        "color": "#F8C471",
                    },
                    {
                        "range": [60, 80],
                        "color": "#F9E79F",
                    },
                    {
                        "range": [80, 100],
                        "color": "#ABEBC6",
                    },
                ],
            },
        )
    )

    health_gauge.update_layout(
        height=350
    )

    st.plotly_chart(
        health_gauge,
        use_container_width=True,
    )


with gauge_column_2:
    risk_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=selected_asset[
                "risk_score"
            ],
            title={
                "text": "Maintenance Risk Score"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "steps": [
                    {
                        "range": [0, 30],
                        "color": "#ABEBC6",
                    },
                    {
                        "range": [30, 60],
                        "color": "#F9E79F",
                    },
                    {
                        "range": [60, 80],
                        "color": "#F8C471",
                    },
                    {
                        "range": [80, 100],
                        "color": "#F5B7B1",
                    },
                ],
            },
        )
    )

    risk_gauge.update_layout(
        height=350
    )

    st.plotly_chart(
        risk_gauge,
        use_container_width=True,
    )


st.divider()

status_column_1, status_column_2 = st.columns(2)

with status_column_1:
    st.subheader("Asset Status")

    st.markdown(
        f"""
        **Health Status:**  
        {selected_asset['health_status']}

        **Risk Level:**  
        {selected_asset['risk_level']}

        **Maintenance Priority:**  
        {selected_asset['maintenance_priority']}

        **Recommended Window:**  
        {selected_asset['maintenance_window']}
        """
    )

with status_column_2:
    st.subheader("Anomaly Assessment")

    st.markdown(
        f"""
        **Status:**  
        {selected_asset.get(
            'anomaly_status',
            'Not available'
        )}

        **Severity:**  
        {selected_asset.get(
            'anomaly_severity',
            'Not available'
        )}

        **Anomaly Score:**  
        {selected_asset.get(
            'anomaly_score',
            0
        ):.1f}/100
        """
    )


st.subheader("Recommended Maintenance Action")

st.info(
    selected_asset["recommended_action"]
)


st.subheader("Decision Explanation")

st.write(
    selected_asset["decision_explanation"]
)

anomaly_explanation = selected_asset.get(
    "anomaly_explanation"
)

if anomaly_explanation:
    st.write(anomaly_explanation)


st.caption(
    "The maintenance risk score is a rule-based operational "
    "index derived from predicted capped RUL. The anomaly score "
    "measures unusual sensor behaviour relative to the healthy "
    "reference distribution."
)