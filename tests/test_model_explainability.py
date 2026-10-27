import numpy as np
import pandas as pd

from src.model_explainability import (
    create_global_importance,
    normalise_shap_values,
)


def test_normalise_two_dimensional_shap_values():
    shap_values = np.array(
        [
            [1.0, -2.0],
            [0.5, 3.0],
        ]
    )

    result = normalise_shap_values(
        shap_values
    )

    assert result.shape == (2, 2)


def test_normalise_three_dimensional_shap_values():
    shap_values = np.array(
        [
            [
                [1.0],
                [-2.0],
            ],
            [
                [0.5],
                [3.0],
            ],
        ]
    )

    result = normalise_shap_values(
        shap_values
    )

    assert result.shape == (2, 2)


def test_global_importance_ranks_largest_feature_first():
    shap_values = np.array(
        [
            [1.0, 5.0],
            [2.0, -7.0],
            [3.0, 6.0],
        ]
    )

    result = create_global_importance(
        shap_values,
        [
            "feature_a",
            "feature_b",
        ],
    )

    assert (
        result.iloc[0]["feature"]
        == "feature_b"
    )

    assert (
        result.iloc[0]["importance_rank"]
        == 1
    )


def test_global_importance_contains_all_features():
    shap_values = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ]
    )

    result = create_global_importance(
        shap_values,
        [
            "feature_a",
            "feature_b",
            "feature_c",
        ],
    )

    assert len(result) == 3

    assert set(result["feature"]) == {
        "feature_a",
        "feature_b",
        "feature_c",
    }