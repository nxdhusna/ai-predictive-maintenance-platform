AI-Driven Asset Health and Predictive Maintenance Platform
An explainable artificial intelligence platform for monitoring equipment health, predicting Remaining Useful Life, detecting sensor anomalies, and supporting risk-based maintenance decisions.

Project Overview
Unexpected equipment failures can result in operational disruption, unplanned downtime, safety risks, and significant financial losses.

This project aims to develop an AI-powered decision-support platform that processes operational and sensor data to:

Predict equipment Remaining Useful Life
Estimate failure risk
Detect abnormal sensor behaviour
Calculate equipment health scores
Prioritise maintenance actions
Explain model predictions
Estimate potential maintenance cost impact
Target Industries
Aviation
Automotive
Electronics
Manufacturing
Industrial engineering
Transportation and logistics
Planned Technologies
Python
Streamlit
Pandas
NumPy
Scikit-learn
XGBoost
SHAP
Plotly
GitHub
Planned Platform Modules
Executive Overview
Fleet Health Monitor
Asset Diagnostics
Maintenance Priority Queue
Batch Prediction
Model Performance
Cost Impact Simulator
Current Development Status
The project structure and initial Streamlit interface have been created.

Dataset
The machine-learning pipeline will initially use the NASA C-MAPSS turbofan engine degradation dataset.

Disclaimer
This application is a portfolio demonstration developed using benchmark or simulated data. It is not validated for real-world, safety-critical maintenance decisions.


Dataset
The initial modelling pipeline uses the NASA C-MAPSS FD001 turbofan engine degradation dataset.

FD001 contains:

100 training engine trajectories
100 test engine trajectories
One operating condition
One simulated fault mode
Three operational settings
Twenty-one sensor measurements
The training trajectories continue until simulated failure. Test trajectories end before failure, and separate Remaining Useful Life values are provided for evaluation.

Dataset Attribution
Saxena, A. and Goebel, K. (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, California.

Data Pipeline
The data pipeline:

Loads the original space-separated text files.
Applies a consistent 26-column schema.
Checks missing values and duplicated observations.
Validates engine identifiers and operating-cycle sequences.
Generates Remaining Useful Life targets.
Saves processed data for model development.

![img_1.png](img_1.png)

## Time-Aware Feature Engineering

The feature-engineering pipeline creates degradation features while
maintaining engine boundaries and chronological order.

Generated features include:

- Capped Remaining Useful Life target
- Sensor rolling means over 5, 10, and 20 cycles
- Sensor rolling standard deviations
- Previous-cycle sensor differences
- Log-transformed cycle count
- Observed lifecycle fraction

All rolling and difference features are calculated independently for
each engine. This prevents information from one engine trajectory from
entering another engine's feature history.

The final model evaluation will use engine-grouped data splitting rather
than random row splitting to reduce information leakage.

## Baseline RUL Model Training

The project compares multiple regression models for Remaining Useful
Life estimation:

- Dummy mean baseline
- Linear Regression
- Random Forest Regressor
- Histogram Gradient Boosting Regressor

Training and validation data are separated by engine using grouped
splitting. All operating cycles belonging to one engine remain within
a single split, reducing the risk of engine-level information leakage.

Models are evaluated using:

- Mean Absolute Error
- Root Mean Squared Error
- R-squared
- Terminal-cycle prediction performance

The model with the lowest grouped-validation RMSE is saved for
application integration.

The deployed model excludes target variables, engine identifiers, and
features that rely on knowledge of the final observed trajectory.


### Baseline Model Results

Four regression models were evaluated using an engine-grouped
validation split to prevent cycles from the same engine from appearing
in both training and validation data.

The Random Forest Regressor achieved the best validation performance:

- MAE: 9.68 cycles
- RMSE: 14.04 cycles
- R²: 0.887

Compared with the dummy mean baseline, the selected model reduced MAE
by approximately 73.8% and RMSE by approximately 66.3%.

These results represent grouped validation performance. Separate
evaluation against the official FD001 test-engine endpoints is
performed before deployment.


## Official FD001 Test Evaluation

The selected Random Forest model is evaluated against the separate
FD001 test set containing 100 previously unseen engine trajectories.

One prediction is generated at the final observed operating cycle of
each test engine. Predictions are compared against the supplied
Remaining Useful Life targets.

Because the model is trained using an RUL cap of 125 cycles, capped-RUL
metrics are treated as the primary model evaluation. Raw-RUL metrics
are also reported for transparency.

The evaluation includes:

- MAE, RMSE, and R-squared
- Actual versus predicted RUL
- Prediction residual analysis
- Maximum and median absolute error
- Identification of the worst-performing test engines



## Asset Health and Maintenance Intelligence

The platform converts predicted Remaining Useful Life into
decision-support indicators that are easier for operational users to
interpret.

Generated indicators include:

- Asset Health Score from 0 to 100
- Maintenance Risk Score from 0 to 100
- Health status
- Risk classification
- Maintenance priority
- Suggested maintenance window
- Recommended operational action
- Fleet-wide maintenance priority rank

The maintenance risk score is a transparent operational index derived
from predicted capped RUL. It is not presented as a calibrated
probability of equipment failure.

### Maintenance Priority Levels

- P1 - Immediate: Predicted RUL of 10 cycles or fewer
- P2 - High: Predicted RUL between 11 and 30 cycles
- P3 - Scheduled: Predicted RUL between 31 and 60 cycles
- P4 - Routine: Predicted RUL above 60 cycles

The thresholds are prototype decision rules and would require domain
expert validation before use in a real operational environment.



## Failure-Within-30-Cycles Classification

A separate binary classification pipeline estimates whether an engine
has entered the final 30 operating cycles before simulated failure.

Target definition:

- Class 1: Actual RUL is 30 cycles or fewer
- Class 0: Actual RUL is greater than 30 cycles

The classifier uses three mutually exclusive groups of engines:

1. Base-model training engines
2. Probability-calibration engines
3. Final validation engines

Candidate models include Logistic Regression, Random Forest, and
Histogram Gradient Boosting. The selected classifier is calibrated
using sigmoid logistic calibration on a separate set of engine
trajectories.

Evaluation metrics include:

- Precision
- Recall
- F1 score
- ROC-AUC
- Average Precision
- Brier score
- Log loss
- Probability calibration curve

The resulting probability represents the model's estimated probability
that a simulated engine observation lies within the final 30-cycle
failure horizon. It is not validated for real aviation or
safety-critical maintenance decisions.


## Sensor Anomaly Detection

The platform uses an Isolation Forest model to identify unusual
multivariate sensor patterns without requiring explicit anomaly labels.

The anomaly detector is trained using observations with more than
60 cycles of Remaining Useful Life as the healthy reference region.

The anomaly-detection pipeline:

1. Selects sensor-related raw and time-aware features.
2. Applies robust feature scaling.
3. Learns a healthy multivariate reference using Isolation Forest.
4. Generates an anomaly decision score.
5. Converts decision scores into a percentile-based anomaly score.
6. Assigns normal, moderate, high, or critical severity.
7. Analyses anomaly rates across Remaining Useful Life bands.
8. Ranks test assets based on final-cycle anomaly scores.

The anomaly score ranges from 0 to 100 and reflects how unusual an
observation is compared with the healthy reference distribution.

The anomaly score is not a calibrated probability of failure, and an
anomaly does not by itself confirm a physical equipment fault.



## Explainable AI

The platform uses SHAP TreeExplainer to explain the selected Random
Forest RUL model.

### Global Explainability

Fleet-level feature importance is calculated using the mean absolute
SHAP contribution across a reproducible sample of test observations.
This identifies the features with the greatest overall influence on
predicted Remaining Useful Life.

### Asset-Level Explainability

A local explanation is generated for the final observed cycle of each
test engine. The explanation shows:

- The model's baseline RUL prediction
- The final asset-specific RUL prediction
- Features that increase predicted RUL
- Features that reduce predicted RUL
- The magnitude of each contribution

SHAP values explain the behaviour of the fitted model. They do not
establish that a sensor feature physically caused engine degradation.
Correlated rolling and raw sensor features may also share or
redistribute model attribution.

## Maintenance Cost-Impact Simulator

The dashboard includes an interactive scenario-planning module that
translates model probabilities into risk-adjusted financial estimates.

Users can configure:

- Planned maintenance cost per asset
- Unplanned repair cost
- Downtime hours
- Cost per downtime hour
- Estimated avoidable-cost percentage
- Number of high-priority assets evaluated

The simulator calculates:

- Direct unplanned failure impact
- Risk-adjusted failure cost
- Expected avoidable cost
- Net estimated maintenance benefit
- Benefit-cost ratio
- Scenario return on investment
- Potentially avoidable downtime

All financial values are user-defined assumptions. The calculations are
illustrative portfolio scenarios and do not represent guaranteed
savings, financial advice, or validated operational estimates.