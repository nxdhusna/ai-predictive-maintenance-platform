import numpy as np
import pandas as pd

from src.anomaly_detection import (
    assign_anomaly_severity,
    assign_anomaly_status,
    convert_to_anomaly_score,
    create_anomaly_rate_summary,
    select_healthy_reference_data,
)


def create_test_dataframe():
    return pd.DataFrame(
        {
            "engine_id": [
                1,
                1,
                1,
                2,
                2,
                2,
            ],
            "cycle": [
                1,
                2,
                3,
                1,
                2,
                3,
            ],
            "rul": [
                100,
                50,
                10,
                90,
                40,
                5,
            ],
            "is_anomaly": [
                0,
                0,
                1,
                0,
                1,
                1,
            ],
            "anomaly_score": [
                10.0,
                40.0,
                95.0,
                20.0,
                80.0,
                99.0,
            ],
        }
    )


def test_healthy_reference_uses_high_rul_only():
    dataframe = create_test_dataframe()

    healthy_dataframe = (
        select_healthy_reference_data(
            dataframe
        )
    )

    assert (
        healthy_dataframe["rul"] > 60
    ).all()

    assert len(healthy_dataframe) == 2


def test_anomaly_score_is_inverse_percentile():
    healthy_scores = np.array(
        [
            -0.20,
            -0.10,
            0.00,
            0.10,
            0.20,
        ]
    )

    decision_scores = np.array(
        [
            -0.20,
            0.00,
            0.20,
        ]
    )

    anomaly_scores = (
        convert_to_anomaly_score(
            decision_scores,
            healthy_scores,
        )
    )

    assert anomaly_scores[0] > anomaly_scores[1]
    assert anomaly_scores[1] > anomaly_scores[2]


def test_anomaly_scores_remain_bounded():
    healthy_scores = np.array(
        [
            -0.10,
            0.00,
            0.10,
        ]
    )

    decision_scores = np.array(
        [
            -10.0,
            0.0,
            10.0,
        ]
    )

    anomaly_scores = (
        convert_to_anomaly_score(
            decision_scores,
            healthy_scores,
        )
    )

    assert (
        anomaly_scores >= 0
    ).all()

    assert (
        anomaly_scores <= 100
    ).all()


def test_anomaly_severity_thresholds():
    assert (
        assign_anomaly_severity(98)
        == "Critical"
    )

    assert (
        assign_anomaly_severity(85)
        == "High"
    )

    assert (
        assign_anomaly_severity(65)
        == "Moderate"
    )

    assert (
        assign_anomaly_severity(30)
        == "Normal"
    )


def test_anomaly_status_conversion():
    assert (
        assign_anomaly_status(-1)
        == "Anomaly Detected"
    )

    assert (
        assign_anomaly_status(1)
        == "Normal Behaviour"
    )


def test_anomaly_rate_summary_is_created():
    dataframe = create_test_dataframe()

    summary = create_anomaly_rate_summary(
        dataframe
    )

    assert "anomaly_rate" in summary.columns

    assert (
        summary["anomaly_rate"]
        .dropna()
        .between(
            0,
            100,
        )
        .all()
    )