import numpy as np
import pandas as pd


DEFAULT_AVOIDABLE_COST_RATE = 0.70


def calculate_failure_impact(
    unplanned_repair_cost,
    downtime_hours,
    downtime_cost_per_hour,
):
    """
    Calculate the direct financial impact of one unplanned
    equipment failure.
    """

    repair_cost = max(
        float(unplanned_repair_cost),
        0.0,
    )

    downtime_hours = max(
        float(downtime_hours),
        0.0,
    )

    hourly_downtime_cost = max(
        float(downtime_cost_per_hour),
        0.0,
    )

    downtime_cost = (
        downtime_hours
        * hourly_downtime_cost
    )

    total_failure_impact = (
        repair_cost
        + downtime_cost
    )

    return {
        "unplanned_repair_cost": repair_cost,
        "downtime_hours": downtime_hours,
        "downtime_cost_per_hour": (
            hourly_downtime_cost
        ),
        "downtime_cost": downtime_cost,
        "total_failure_impact": (
            total_failure_impact
        ),
    }


def normalise_probability(probability):
    """
    Convert a percentage or decimal probability to a value
    between zero and one.

    Examples:
    0.75 becomes 0.75.
    75 becomes 0.75.
    """

    probability = float(probability)

    if probability > 1:
        probability = (
            probability / 100
        )

    return float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )


def calculate_asset_cost_scenario(
    failure_probability,
    planned_maintenance_cost,
    unplanned_repair_cost,
    downtime_hours,
    downtime_cost_per_hour,
    avoidable_cost_rate=(
        DEFAULT_AVOIDABLE_COST_RATE
    ),
):
    """
    Calculate one asset's risk-adjusted maintenance scenario.

    The calculation is a financial planning estimate and does
    not represent guaranteed savings.
    """

    probability = normalise_probability(
        failure_probability
    )

    planned_cost = max(
        float(planned_maintenance_cost),
        0.0,
    )

    avoidable_rate = normalise_probability(
        avoidable_cost_rate
    )

    failure_impact = calculate_failure_impact(
        unplanned_repair_cost,
        downtime_hours,
        downtime_cost_per_hour,
    )

    risk_adjusted_failure_cost = (
        failure_impact[
            "total_failure_impact"
        ]
        * probability
    )

    expected_avoidable_cost = (
        risk_adjusted_failure_cost
        * avoidable_rate
    )

    net_estimated_benefit = (
        expected_avoidable_cost
        - planned_cost
    )

    if planned_cost > 0:
        maintenance_roi_percent = (
            net_estimated_benefit
            / planned_cost
        ) * 100

    else:
        maintenance_roi_percent = 0.0

    if expected_avoidable_cost > 0:
        benefit_cost_ratio = (
            expected_avoidable_cost
            / planned_cost
            if planned_cost > 0
            else 0.0
        )

    else:
        benefit_cost_ratio = 0.0

    expected_downtime_hours = (
        failure_impact[
            "downtime_hours"
        ]
        * probability
    )

    potentially_avoidable_downtime = (
        expected_downtime_hours
        * avoidable_rate
    )

    return {
        "failure_probability": probability,
        "planned_maintenance_cost": (
            planned_cost
        ),
        "unplanned_repair_cost": (
            failure_impact[
                "unplanned_repair_cost"
            ]
        ),
        "downtime_hours": (
            failure_impact[
                "downtime_hours"
            ]
        ),
        "downtime_cost_per_hour": (
            failure_impact[
                "downtime_cost_per_hour"
            ]
        ),
        "downtime_cost": (
            failure_impact[
                "downtime_cost"
            ]
        ),
        "total_failure_impact": (
            failure_impact[
                "total_failure_impact"
            ]
        ),
        "risk_adjusted_failure_cost": (
            risk_adjusted_failure_cost
        ),
        "avoidable_cost_rate": (
            avoidable_rate
        ),
        "expected_avoidable_cost": (
            expected_avoidable_cost
        ),
        "net_estimated_benefit": (
            net_estimated_benefit
        ),
        "maintenance_roi_percent": (
            maintenance_roi_percent
        ),
        "benefit_cost_ratio": (
            benefit_cost_ratio
        ),
        "expected_downtime_hours": (
            expected_downtime_hours
        ),
        "potentially_avoidable_downtime": (
            potentially_avoidable_downtime
        ),
    }


def calculate_fleet_cost_scenario(
    asset_dataframe,
    probability_column,
    planned_maintenance_cost,
    unplanned_repair_cost,
    downtime_hours,
    downtime_cost_per_hour,
    avoidable_cost_rate=(
        DEFAULT_AVOIDABLE_COST_RATE
    ),
):
    """
    Calculate cost-impact scenarios for multiple assets.
    """

    if asset_dataframe.empty:
        raise ValueError(
            "The selected fleet data is empty."
        )

    if (
        probability_column
        not in asset_dataframe.columns
    ):
        raise ValueError(
            "Probability column was not found: "
            f"{probability_column}"
        )

    result_records = []

    for _, asset_row in (
        asset_dataframe.iterrows()
    ):
        scenario = calculate_asset_cost_scenario(
            failure_probability=(
                asset_row[
                    probability_column
                ]
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
                avoidable_cost_rate
            ),
        )

        record = {
            "engine_id": int(
                asset_row["engine_id"]
            ),
            "predicted_rul": float(
                asset_row.get(
                    "predicted_rul",
                    np.nan,
                )
            ),
            "risk_level": asset_row.get(
                "risk_level",
                "Not available",
            ),
            "maintenance_priority": (
                asset_row.get(
                    "maintenance_priority",
                    "Not available",
                )
            ),
            **scenario,
        }

        result_records.append(
            record
        )

    result = pd.DataFrame(
        result_records
    )

    result = result.sort_values(
        by="net_estimated_benefit",
        ascending=False,
    ).reset_index(drop=True)

    result[
        "financial_priority_rank"
    ] = range(
        1,
        len(result) + 1,
    )

    return result


def create_fleet_cost_summary(
    fleet_cost_dataframe,
):
    """
    Summarise the fleet-level financial scenario.
    """

    if fleet_cost_dataframe.empty:
        raise ValueError(
            "Fleet cost scenario data is empty."
        )

    total_planned_cost = float(
        fleet_cost_dataframe[
            "planned_maintenance_cost"
        ].sum()
    )

    total_risk_adjusted_failure_cost = float(
        fleet_cost_dataframe[
            "risk_adjusted_failure_cost"
        ].sum()
    )

    total_expected_avoidable_cost = float(
        fleet_cost_dataframe[
            "expected_avoidable_cost"
        ].sum()
    )

    total_net_estimated_benefit = float(
        fleet_cost_dataframe[
            "net_estimated_benefit"
        ].sum()
    )

    total_avoidable_downtime = float(
        fleet_cost_dataframe[
            "potentially_avoidable_downtime"
        ].sum()
    )

    if total_planned_cost > 0:
        fleet_roi_percent = (
            total_net_estimated_benefit
            / total_planned_cost
        ) * 100

        fleet_benefit_cost_ratio = (
            total_expected_avoidable_cost
            / total_planned_cost
        )

    else:
        fleet_roi_percent = 0.0
        fleet_benefit_cost_ratio = 0.0

    positive_business_cases = int(
        (
            fleet_cost_dataframe[
                "net_estimated_benefit"
            ]
            > 0
        ).sum()
    )

    return {
        "assets_evaluated": int(
            len(fleet_cost_dataframe)
        ),
        "total_planned_maintenance_cost": (
            total_planned_cost
        ),
        "total_risk_adjusted_failure_cost": (
            total_risk_adjusted_failure_cost
        ),
        "total_expected_avoidable_cost": (
            total_expected_avoidable_cost
        ),
        "total_net_estimated_benefit": (
            total_net_estimated_benefit
        ),
        "total_potentially_avoidable_downtime": (
            total_avoidable_downtime
        ),
        "fleet_roi_percent": (
            fleet_roi_percent
        ),
        "fleet_benefit_cost_ratio": (
            fleet_benefit_cost_ratio
        ),
        "positive_business_cases": (
            positive_business_cases
        ),
    }