import plotly.express as px
import streamlit as st

from src.dashboard_utils import (
    load_explainability_metadata,
    load_global_feature_importance,
    load_local_asset_explanations,
    load_local_explanation_summary,
    render_project_disclaimer,
    render_sidebar_information,
)


st.set_page_config(
    page_title="Explainable AI",
    page_icon="🧠",
    layout="wide",
)


render_sidebar_information()

st.title("Explainable AI")

st.markdown(
    """
    Understand which model features influence Remaining Useful
    Life predictions across the fleet and for individual assets.
    """
)

render_project_disclaimer()


try:
    global_importance = (
        load_global_feature_importance()
    )

    local_explanations = (
        load_local_asset_explanations()
    )

    local_summaries = (
        load_local_explanation_summary()
    )

    explanation_metadata = (
        load_explainability_metadata()
    )

except FileNotFoundError as error:
    st.error(str(error))

    st.info(
        "Run 'python -m src.model_explainability' "
        "before opening this page."
    )

    st.stop()


st.subheader("How to Read the Explanation")

explanation_1, explanation_2, explanation_3 = (
    st.columns(3)
)

with explanation_1:
    st.markdown("#### Positive Contribution")

    st.write(
        "A positive SHAP value pushes the model toward a "
        "higher Remaining Useful Life prediction."
    )

with explanation_2:
    st.markdown("#### Negative Contribution")

    st.write(
        "A negative SHAP value pushes the model toward a "
        "lower Remaining Useful Life prediction."
    )

with explanation_3:
    st.markdown("#### Influence Magnitude")

    st.write(
        "The absolute SHAP value indicates how strongly a "
        "feature affects the prediction."
    )


st.divider()

st.subheader("Global Model Explainability")

top_feature_count = st.slider(
    label="Number of features to display",
    min_value=5,
    max_value=min(
        30,
        len(global_importance),
    ),
    value=min(
        15,
        len(global_importance),
    ),
)


global_plot_data = (
    global_importance
    .head(top_feature_count)
    .sort_values(
        "mean_absolute_shap",
        ascending=True,
    )
)


global_chart = px.bar(
    global_plot_data,
    x="mean_absolute_shap",
    y="feature",
    orientation="h",
    color="mean_signed_shap",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    title=(
        "Global Feature Influence on "
        "Predicted Remaining Useful Life"
    ),
    labels={
        "mean_absolute_shap": (
            "Mean Absolute SHAP Contribution "
            "in RUL Cycles"
        ),
        "feature": "Model Feature",
        "mean_signed_shap": (
            "Mean Signed Contribution"
        ),
    },
)

st.plotly_chart(
    global_chart,
    use_container_width=True,
)


st.caption(
    "Global importance is calculated from the mean absolute "
    "SHAP value. The result reflects model influence, not a "
    "causal relationship with real engine degradation."
)


st.divider()

st.subheader("Individual Asset Explanation")

engine_ids = sorted(
    local_summaries[
        "engine_id"
    ].astype(int).unique()
)


selected_engine_id = st.selectbox(
    label="Select Engine",
    options=engine_ids,
    format_func=lambda engine_id: (
        f"Engine {engine_id}"
    ),
)


selected_summary = local_summaries.loc[
    local_summaries["engine_id"]
    == selected_engine_id
].iloc[0]


selected_contributions = (
    local_explanations.loc[
        local_explanations["engine_id"]
        == selected_engine_id
    ]
    .sort_values(
        "absolute_shap_value",
        ascending=False,
    )
)


metric_1, metric_2, metric_3 = st.columns(3)

with metric_1:
    st.metric(
        "Baseline RUL",
        (
            f"{selected_summary[
                'baseline_rul'
            ]:.1f} cycles"
        ),
    )

with metric_2:
    st.metric(
        "Predicted RUL",
        (
            f"{selected_summary[
                'predicted_rul'
            ]:.1f} cycles"
        ),
    )

with metric_3:
    prediction_difference = (
        selected_summary["predicted_rul"]
        - selected_summary["baseline_rul"]
    )

    st.metric(
        "Difference from Baseline",
        f"{prediction_difference:+.1f} cycles",
    )


top_local_feature_count = st.slider(
    label="Number of local contributions",
    min_value=5,
    max_value=min(
        20,
        len(selected_contributions),
    ),
    value=min(
        12,
        len(selected_contributions),
    ),
)


local_plot_data = (
    selected_contributions
    .head(top_local_feature_count)
    .sort_values(
        "shap_value",
        ascending=True,
    )
    .copy()
)


local_plot_data["Contribution Type"] = (
    local_plot_data["shap_value"]
    .apply(
        lambda value: (
            "Increases RUL"
            if value > 0
            else "Reduces RUL"
        )
    )
)


local_chart = px.bar(
    local_plot_data,
    x="shap_value",
    y="feature",
    orientation="h",
    color="Contribution Type",
    color_discrete_map={
        "Increases RUL": "#2E8B57",
        "Reduces RUL": "#C0392B",
    },
    title=(
        f"Feature Contributions for "
        f"Engine {selected_engine_id}"
    ),
    labels={
        "shap_value": (
            "Contribution to Predicted RUL Cycles"
        ),
        "feature": "Model Feature",
    },
)

local_chart.add_vline(
    x=0,
    line_dash="dash",
    line_color="black",
)

st.plotly_chart(
    local_chart,
    use_container_width=True,
)


explanation_column_1, explanation_column_2 = (
    st.columns(2)
)

with explanation_column_1:
    st.markdown(
        "#### Strongest RUL-Reducing Features"
    )

    reducing_features = (
        selected_contributions.loc[
            selected_contributions[
                "shap_value"
            ] < 0
        ]
        .head(5)
    )

    if reducing_features.empty:
        st.write(
            "No negative feature contributions "
            "were identified."
        )

    else:
        for _, row in reducing_features.iterrows():
            st.write(
                f"- **{row['feature']}**: "
                f"{row['shap_value']:.2f} cycles"
            )


with explanation_column_2:
    st.markdown(
        "#### Strongest RUL-Increasing Features"
    )

    increasing_features = (
        selected_contributions.loc[
            selected_contributions[
                "shap_value"
            ] > 0
        ]
        .head(5)
    )

    if increasing_features.empty:
        st.write(
            "No positive feature contributions "
            "were identified."
        )

    else:
        for _, row in increasing_features.iterrows():
            st.write(
                f"- **{row['feature']}**: "
                f"+{row['shap_value']:.2f} cycles"
            )


st.subheader("Feature Contribution Data")

display_dataframe = (
    selected_contributions[
        [
            "contribution_rank",
            "feature",
            "feature_value",
            "shap_value",
            "contribution_direction",
        ]
    ]
    .head(top_local_feature_count)
    .copy()
)


display_dataframe = display_dataframe.rename(
    columns={
        "contribution_rank": "Rank",
        "feature": "Feature",
        "feature_value": "Feature Value",
        "shap_value": "RUL Contribution",
        "contribution_direction": "Direction",
    }
)


st.dataframe(
    display_dataframe,
    use_container_width=True,
    hide_index=True,
)


st.info(
    "SHAP explains how the trained model reached a prediction. "
    "Feature contributions should not be interpreted as proof "
    "that a particular sensor caused physical degradation."
)