import numpy as np
import pandas as pd

from src.maintenance_intelligence import (
    assign_health_status,
    assign_maintenance_priority,
    assign_maintenance_window,
    assign_risk_level,
    build_maintenance_intelligence,
    calculate_health_score,
    calculate_risk_score,
)


def create_prediction_dataframe():
    return pd.DataFrame(
        {
            "engine_id": [
                1,
                2,
                3,
                4,
            ],
            "final_observed_cycle": [
                100,
                120,
                150,
                180,
            ],
            "actual_rul": [
                5,
                25,
                55,
                100,
            ],
            "actual_rul_capped": [
                5,
                25,
                55,
                100,
            ],
            "predicted_rul": [
                5.0,
                25.0,
                55.0,
                110.0,
            ],
        }
    )


def test_health_score_boundaries():
    predicted_rul = np.array(
        [
            0.0,
            62.5,
            125.0,
            150.0,
        ]
    )

    health_scores = calculate_health_score(
        predicted_rul
    )

    assert np.allclose(
        health_scores,
        [
            0.0,
            50.0,
            100.0,
            100.0,
        ],
    )


def test_risk_score_is_inverse_of_health():
    health_scores = np.array(
        [
            0.0,
            40.0,
            75.0,
            100.0,
        ]
    )

    risk_scores = calculate_risk_score(
        health_scores
    )

    assert np.allclose(
        risk_scores,
        [
            100.0,
            60.0,
            25.0,
            0.0,
        ],
    )


def test_health_status_thresholds():
    assert assign_health_status(90) == "Healthy"
    assert assign_health_status(70) == "Monitor"

    assert (
        assign_health_status(50)
        == "Maintenance Required"
    )

    assert assign_health_status(20) == "Critical"


def test_risk_level_thresholds():
    assert assign_risk_level(90) == "Critical"
    assert assign_risk_level(70) == "High"
    assert assign_risk_level(40) == "Moderate"
    assert assign_risk_level(20) == "Low"


def test_maintenance_priority_thresholds():
    assert (
        assign_maintenance_priority(8)
        == "P1 - Immediate"
    )

    assert (
        assign_maintenance_priority(20)
        == "P2 - High"
    )

    assert (
        assign_maintenance_priority(45)
        == "P3 - Scheduled"
    )

    assert (
        assign_maintenance_priority(90)
        == "P4 - Routine"
    )


def test_maintenance_window_thresholds():
    assert (
        assign_maintenance_window(5)
        == "Before the next 5 operating cycles"
    )

    assert (
        assign_maintenance_window(20)
        == "Within the next 10 operating cycles"
    )

    assert (
        assign_maintenance_window(50)
        == "Within the next 30 operating cycles"
    )

    assert (
        assign_maintenance_window(100)
        == "Continue routine monitoring"
    )


def test_priority_queue_places_lowest_rul_first():
    dataframe = create_prediction_dataframe()

    result = build_maintenance_intelligence(
        dataframe
    )

    first_engine = int(
        result.iloc[0]["engine_id"]
    )

    assert first_engine == 1


def test_maintenance_scores_remain_bounded():
    dataframe = create_prediction_dataframe()

    result = build_maintenance_intelligence(
        dataframe
    )

    assert result["health_score"].between(
        0,
        100,
    ).all()

    assert result["risk_score"].between(
        0,
        100,
    ).all()