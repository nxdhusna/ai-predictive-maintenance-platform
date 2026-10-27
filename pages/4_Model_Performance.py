import plotly.express as px
import streamlit as st

from src.dashboard_utils import (
    load_failure_metrics,
    load_model_comparison,
    render_project_disclaimer,
    render_sidebar_information,
)


st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide",
)


render_sidebar_information()

st.title("Model Performance")

st.markdown(
    """
    Review grouped-validation regression performance and
    calibrated failure-classification results.
    """
)

render_project_disclaimer()


try:
    model_comparison = load_model_comparison()

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


best_model = (
    model_comparison
    .sort_values("rmse")
    .iloc[0]
)


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

with metric_1:
    st.metric(
        "Selected RUL Model",
        best_model["model"],
    )

with metric_2:
    st.metric(
        "Validation MAE",
        f"{best_model['mae']:.2f} cycles",
    )

with metric_3:
    st.metric(
        "Validation RMSE",
        f"{best_model['rmse']:.2f} cycles",
    )

with metric_4:
    st.metric(
        "Validation R²",
        f"{best_model['r2']:.3f}",
    )


comparison_long = model_comparison.melt(
    id_vars=["model"],
    value_vars=["mae", "rmse"],
    var_name="Metric",
    value_name="Error",
)

regression_chart = px.bar(
    comparison_long,
    x="model",
    y="Error",
    color="Metric",
    barmode="group",
    title="RUL Model Error Comparison",
    labels={
        "model": "Model",
        "Error": "Error in RUL Cycles",
    },
)

st.plotly_chart(
    regression_chart,
    use_container_width=True,
)


st.subheader("Regression Model Results")

st.dataframe(
    model_comparison,
    use_container_width=True,
    hide_index=True,
)


st.divider()

st.subheader(
    "Failure-Within-30-Cycles Classifier"
)


try:
    failure_metrics = load_failure_metrics()

    validation_metrics = failure_metrics[
        "validation_metrics"
    ]

    classifier_1, classifier_2, classifier_3, classifier_4 = (
        st.columns(4)
    )

    with classifier_1:
        st.metric(
            "Precision",
            f"{validation_metrics['precision']:.3f}",
        )

    with classifier_2:
        st.metric(
            "Recall",
            f"{validation_metrics['recall']:.3f}",
        )

    with classifier_3:
        st.metric(
            "F1 Score",
            f"{validation_metrics['f1']:.3f}",
        )

    with classifier_4:
        st.metric(
            "ROC-AUC",
            f"{validation_metrics['roc_auc']:.3f}",
        )


    probability_1, probability_2 = st.columns(2)

    with probability_1:
        st.metric(
            "Average Precision",
            (
                f"{validation_metrics[
                    'average_precision'
                ]:.3f}"
            ),
        )

    with probability_2:
        st.metric(
            "Brier Score",
            (
                f"{validation_metrics[
                    'brier_score'
                ]:.4f}"
            ),
        )

    st.caption(
        "Lower Brier score indicates smaller probability "
        "prediction error. Probability calibration should also "
        "be assessed using the saved calibration curve."
    )

except FileNotFoundError:
    st.info(
        "Failure-classification results are not yet available. "
        "Run 'python -m src.failure_classifier'."
    )


st.divider()

st.subheader("Evaluation Methodology")

st.markdown(
    """
    - Engine trajectories were separated by engine identifier.
    - Cycles from one engine were not distributed across training
      and validation sets.
    - The RUL model was evaluated using MAE, RMSE, and R².
    - The failure classifier used separate training, calibration,
      and validation engine groups.
    - The deployed outputs are developed using simulated benchmark
      data and are not validated for operational use.
    """
)