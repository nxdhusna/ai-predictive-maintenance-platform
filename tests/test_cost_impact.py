import pandas as pd

from src.cost_impact import (
    calculate_asset_cost_scenario,
    calculate_failure_impact,
    calculate_fleet_cost_scenario,
    create_fleet_cost_summary,
    normalise_probability,
)


def test_failure_impact_calculation():
    result = calculate_failure_impact(
        unplanned_repair_cost=100000,
        downtime_hours=20,
        downtime_cost_per_hour=5000,
    )

    assert result["downtime_cost"] == 100000

    assert (
        result["total_failure_impact"]
        == 200000
    )


def test_probability_normalisation():
    assert normalise_probability(75) == 0.75
    assert normalise_probability(0.75) == 0.75
    assert normalise_probability(150) == 1.0
    assert normalise_probability(-20) == 0.0


def test_asset_cost_scenario():
    result = calculate_asset_cost_scenario(
        failure_probability=0.50,
        planned_maintenance_cost=20000,
        unplanned_repair_cost=100000,
        downtime_hours=20,
        downtime_cost_per_hour=5000,
        avoidable_cost_rate=0.70,
    )

    assert (
        result["total_failure_impact"]
        == 200000
    )

    assert (
        result["risk_adjusted_failure_cost"]
        == 100000
    )

    assert (
        result["expected_avoidable_cost"]
        == 70000
    )

    assert (
        result["net_estimated_benefit"]
        == 50000
    )


def test_fleet_cost_scenario():
    dataframe = pd.DataFrame(
        {
            "engine_id": [1, 2],
            "predicted_rul": [10, 20],
            "failure_probability": [
                0.80,
                0.40,
            ],
            "risk_level": [
                "Critical",
                "High",
            ],
            "maintenance_priority": [
                "P1 - Immediate",
                "P2 - High",
            ],
        }
    )

    result = calculate_fleet_cost_scenario(
        asset_dataframe=dataframe,
        probability_column=(
            "failure_probability"
        ),
        planned_maintenance_cost=20000,
        unplanned_repair_cost=100000,
        downtime_hours=10,
        downtime_cost_per_hour=5000,
        avoidable_cost_rate=0.70,
    )

    assert len(result) == 2

    assert (
        result["financial_priority_rank"]
        .tolist()
        == [1, 2]
    )


def test_fleet_summary():
    dataframe = pd.DataFrame(
        {
            "planned_maintenance_cost": [
                20000,
                20000,
            ],
            "risk_adjusted_failure_cost": [
                100000,
                50000,
            ],
            "expected_avoidable_cost": [
                70000,
                35000,
            ],
            "net_estimated_benefit": [
                50000,
                15000,
            ],
            "potentially_avoidable_downtime": [
                7,
                3.5,
            ],
        }
    )

    summary = create_fleet_cost_summary(
        dataframe
    )

    assert (
        summary[
            "total_planned_maintenance_cost"
        ]
        == 40000
    )

    assert (
        summary[
            "total_expected_avoidable_cost"
        ]
        == 105000
    )

    assert (
        summary[
            "total_net_estimated_benefit"
        ]
        == 65000
    )

    assert (
        summary["positive_business_cases"]
        == 2
    )