import pandas as pd
import plotly.express as px
import streamlit as st

from src.cost_impact import (
    calculate_fleet_cost_scenario,
    create_fleet_cost_summary,
)
from src.dashboard_utils import (
    combine_asset_and_anomaly_data,
    render_project_disclaimer,
    render_sidebar_information,
)


st.set_page_config(
    page_title="Cost Impact Simulator",
    page_icon="💰",
    layout="wide",
)


render_sidebar_information()

st.title("Maintenance Cost-Impact Simulator")

st.markdown(
    """
    Estimate the potential financial and downtime impact of
    planned maintenance under user-defined business assumptions.
    """
)

render_project_disclaimer()


st.info(
    "All financial values on this page are user-defined scenario "
    "assumptions. Results are illustrative estimates and do not "
    "represent guaranteed cost savings."
)


try:
    asset_dataframe = (
        combine_asset_and_anomaly_data()
    )

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


if (
    "failure_probability"
    not in asset_dataframe.columns
):
    st.error(
        "Official test failure probabilities are unavailable."
    )

    st.code(
        "python -m src.official_failure_scoring"
    )

    st.stop()


st.subheader("Scenario Assumptions")

input_column_1, input_column_2 = st.columns(2)

with input_column_1:
    planned_maintenance_cost = st.number_input(
        label=(
            "Planned maintenance cost "
            "per asset (RM)"
        ),
        min_value=0.0,
        value=25000.0,
        step=1000.0,
    )

    unplanned_repair_cost = st.number_input(
        label=(
            "Unplanned repair cost "
            "per asset (RM)"
        ),
        min_value=0.0,
        value=100000.0,
        step=5000.0,
    )

    downtime_hours = st.number_input(
        label=(
            "Estimated downtime after "
            "failure (hours)"
        ),
        min_value=0.0,
        value=24.0,
        step=1.0,
    )


with input_column_2:
    downtime_cost_per_hour = st.number_input(
        label="Downtime cost per hour (RM)",
        min_value=0.0,
        value=5000.0,
        step=500.0,
    )

    avoidable_cost_percent = st.slider(
        label=(
            "Estimated percentage of failure "
            "impact avoidable through maintenance"
        ),
        min_value=0,
        max_value=100,
        value=70,
        step=5,
    )

    maximum_assets = min(
        100,
        len(asset_dataframe),
    )

    assets_to_evaluate = st.slider(
        label=(
            "Number of highest-priority assets "
            "to evaluate"
        ),
        min_value=1,
        max_value=maximum_assets,
        value=min(
            20,
            maximum_assets,
        ),
    )


selected_assets = (
    asset_dataframe
    .sort_values(
        "fleet_priority_rank"
    )
    .head(assets_to_evaluate)
    .copy()
)


fleet_cost_dataframe = (
    calculate_fleet_cost_scenario(
        asset_dataframe=selected_assets,
        probability_column=(
            "failure_probability"
        ),
        planned_maintenance_cost=(
            planned_maintenance_cost
        ),
        unplanned_repair_cost=(
            unplanned_repair_cost
        ),
        downtime_hours=downtime_hours,
        downtime_cost_per_hour=(
            downtime_cost_per_hour
        ),
        avoidable_cost_rate=(
            avoidable_cost_percent / 100
        ),
    )
)


fleet_summary = create_fleet_cost_summary(
    fleet_cost_dataframe
)


st.divider()

st.subheader("Estimated Fleet Business Impact")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        label="Planned Maintenance Cost",
        value=(
            "RM "
            f"{fleet_summary[
                'total_planned_maintenance_cost'
            ]:,.0f}"
        ),
    )

with metric_2:
    st.metric(
        label="Risk-Adjusted Failure Cost",
        value=(
            "RM "
            f"{fleet_summary[
                'total_risk_adjusted_failure_cost'
            ]:,.0f}"
        ),
    )

with metric_3:
    st.metric(
        label="Expected Avoidable Cost",
        value=(
            "RM "
            f"{fleet_summary[
                'total_expected_avoidable_cost'
            ]:,.0f}"
        ),
    )

with metric_4:
    st.metric(
        label="Net Estimated Benefit",
        value=(
            "RM "
            f"{fleet_summary[
                'total_net_estimated_benefit'
            ]:,.0f}"
        ),
    )


metric_5, metric_6, metric_7, metric_8 = st.columns(4)

with metric_5:
    st.metric(
        label="Scenario ROI",
        value=(
            f"{fleet_summary[
                'fleet_roi_percent'
            ]:,.1f}%"
        ),
    )

with metric_6:
    st.metric(
        label="Benefit-Cost Ratio",
        value=(
            f"{fleet_summary[
                'fleet_benefit_cost_ratio'
            ]:,.2f}"
        ),
    )

with metric_7:
    st.metric(
        label="Potential Downtime Avoided",
        value=(
            f"{fleet_summary[
                'total_potentially_avoidable_downtime'
            ]:,.1f} hours"
        ),
    )

with metric_8:
    st.metric(
        label="Positive Business Cases",
        value=(
            f"{fleet_summary[
                'positive_business_cases'
            ]} of "
            f"{fleet_summary[
                'assets_evaluated'
            ]}"
        ),
    )


st.divider()

st.subheader("Asset-Level Financial Priorities")

chart_data = (
    fleet_cost_dataframe
    .head(20)
    .copy()
)

chart_data["Engine"] = (
    "Engine "
    + chart_data["engine_id"]
    .astype(int)
    .astype(str)
)


benefit_chart = px.bar(
    chart_data,
    x="net_estimated_benefit",
    y="Engine",
    orientation="h",
    color="failure_probability",
    color_continuous_scale="OrRd",
    title=(
        "Estimated Net Maintenance Benefit "
        "by Asset"
    ),
    labels={
        "net_estimated_benefit": (
            "Net Estimated Benefit (RM)"
        ),
        "failure_probability": (
            "Failure Probability"
        ),
    },
)


benefit_chart.update_layout(
    yaxis={
        "categoryorder": "total ascending"
    }
)


st.plotly_chart(
    benefit_chart,
    use_container_width=True,
)


st.subheader("Scenario Details")

display_columns = [
    "financial_priority_rank",
    "engine_id",
    "predicted_rul",
    "failure_probability",
    "planned_maintenance_cost",
    "risk_adjusted_failure_cost",
    "expected_avoidable_cost",
    "net_estimated_benefit",
    "potentially_avoidable_downtime",
]


display_dataframe = fleet_cost_dataframe[
    display_columns
].copy()


display_dataframe[
    "failure_probability"
] = (
    display_dataframe[
        "failure_probability"
    ]
    * 100
)


display_dataframe = display_dataframe.rename(
    columns={
        "financial_priority_rank": (
            "Financial Priority"
        ),
        "engine_id": "Engine ID",
        "predicted_rul": "Predicted RUL",
        "failure_probability": (
            "Failure Probability (%)"
        ),
        "planned_maintenance_cost": (
            "Planned Cost (RM)"
        ),
        "risk_adjusted_failure_cost": (
            "Risk-Adjusted Failure Cost (RM)"
        ),
        "expected_avoidable_cost": (
            "Expected Avoidable Cost (RM)"
        ),
        "net_estimated_benefit": (
            "Net Estimated Benefit (RM)"
        ),
        "potentially_avoidable_downtime": (
            "Potential Downtime Avoided (Hours)"
        ),
    }
)


st.dataframe(
    display_dataframe,
    use_container_width=True,
    hide_index=True,
)


download_data = display_dataframe.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Cost-Impact Scenario",
    data=download_data,
    file_name=(
        "maintenance_cost_impact_scenario.csv"
    ),
    mime="text/csv",
)


st.divider()

st.subheader("Methodology and Limitations")

st.markdown(
    """
    The simulator calculates expected financial impact using:

    - The calibrated probability of entering the final
      30-cycle failure horizon
    - User-defined repair and downtime assumptions
    - A user-defined avoidable-cost percentage
    - Planned maintenance cost per selected asset

    The simulator does not include:

    - Actual component quotations
    - Regulatory compliance costs
    - Safety or environmental consequences
    - Supply-chain delays
    - Spare-parts availability
    - Production dependencies between assets
    - Guaranteed maintenance effectiveness

    The output should be interpreted as a portfolio demonstration
    of risk-adjusted decision support, not as financial advice or
    an operational maintenance commitment.
    """
)