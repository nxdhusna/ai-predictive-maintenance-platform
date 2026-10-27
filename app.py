import streamlit as st


st.set_page_config(
    page_title="AI Predictive Maintenance Platform",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("AI-Driven Asset Health and Predictive Maintenance Platform")

st.markdown(
    """
    An explainable artificial intelligence platform for monitoring 
    equipment health, predicting Remaining Useful Life, detecting 
    sensor anomalies, and supporting risk-based maintenance decisions.
    """
)

st.info(
    "The predictive models and maintenance intelligence features "
    "will be introduced progressively during development."
)

st.subheader("Platform Capabilities")

feature_1, feature_2, feature_3, feature_4 = st.columns(4)

with feature_1:
    st.metric(
        label="Assets Monitored",
        value="100",
        delta="Demo data",
    )

with feature_2:
    st.metric(
        label="Average Health Score",
        value="78/100",
        delta="Healthy",
    )

with feature_3:
    st.metric(
        label="Critical Assets",
        value="9",
        delta="-2 since last inspection",
        delta_color="inverse",
    )

with feature_4:
    st.metric(
        label="Average Predicted RUL",
        value="83 cycles",
        delta="5 cycles",
    )

st.divider()

left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Planned AI Features")

    st.markdown(
        """
        - Remaining Useful Life prediction
        - Failure-risk probability
        - Asset health scoring
        - Sensor anomaly detection
        - Explainable AI analysis
        - Maintenance recommendation engine
        - Maintenance-priority ranking
        - Cost-impact simulation
        """
    )

with right_column:
    st.subheader("Target Industries")

    st.markdown(
        """
        - Aviation
        - Automotive
        - Electronics
        - Manufacturing
        - Industrial engineering
        - Transportation and logistics
        """
    )

st.divider()

st.warning(
    "Portfolio disclaimer: This platform is developed using benchmark "
    "or simulated data. It is not validated for real-world, "
    "safety-critical maintenance decisions."
)