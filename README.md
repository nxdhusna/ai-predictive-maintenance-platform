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
