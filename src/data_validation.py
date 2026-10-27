from dataclasses import dataclass

import pandas as pd

from data_loader import COLUMN_NAMES, load_fd001_data


@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    message: str


def validate_column_count(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> ValidationResult:
    expected_count = len(COLUMN_NAMES)
    actual_count = dataframe.shape[1]

    passed = actual_count == expected_count

    return ValidationResult(
        check_name=f"{dataset_name}: column count",
        passed=passed,
        message=(
            f"Expected {expected_count} columns; "
            f"found {actual_count} columns."
        ),
    )


def validate_column_names(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> ValidationResult:
    actual_columns = list(dataframe.columns)
    passed = actual_columns == COLUMN_NAMES

    return ValidationResult(
        check_name=f"{dataset_name}: column names",
        passed=passed,
        message=(
            "Column names are valid."
            if passed
            else "Columns do not match the expected schema."
        ),
    )


def validate_missing_values(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> ValidationResult:
    missing_count = int(dataframe.isna().sum().sum())
    passed = missing_count == 0

    return ValidationResult(
        check_name=f"{dataset_name}: missing values",
        passed=passed,
        message=f"Found {missing_count} missing values.",
    )


def validate_duplicate_rows(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> ValidationResult:
    duplicate_count = int(dataframe.duplicated().sum())
    passed = duplicate_count == 0

    return ValidationResult(
        check_name=f"{dataset_name}: duplicate rows",
        passed=passed,
        message=f"Found {duplicate_count} duplicate rows.",
    )


def validate_engine_cycle_uniqueness(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> ValidationResult:
    duplicate_cycle_count = int(
        dataframe.duplicated(
            subset=["engine_id", "cycle"]
        ).sum()
    )

    passed = duplicate_cycle_count == 0

    return ValidationResult(
        check_name=f"{dataset_name}: engine-cycle uniqueness",
        passed=passed,
        message=(
            f"Found {duplicate_cycle_count} duplicated "
            "engine-cycle combinations."
        ),
    )


def validate_positive_engine_ids(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> ValidationResult:
    invalid_count = int((dataframe["engine_id"] <= 0).sum())
    passed = invalid_count == 0

    return ValidationResult(
        check_name=f"{dataset_name}: engine identifiers",
        passed=passed,
        message=f"Found {invalid_count} invalid engine identifiers.",
    )


def validate_positive_cycles(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> ValidationResult:
    invalid_count = int((dataframe["cycle"] <= 0).sum())
    passed = invalid_count == 0

    return ValidationResult(
        check_name=f"{dataset_name}: operating cycles",
        passed=passed,
        message=f"Found {invalid_count} invalid cycle values.",
    )


def validate_cycle_sequence(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> ValidationResult:
    invalid_engines = []

    for engine_id, engine_data in dataframe.groupby("engine_id"):
        actual_cycles = engine_data["cycle"].sort_values().tolist()
        expected_cycles = list(
            range(1, int(engine_data["cycle"].max()) + 1)
        )

        if actual_cycles != expected_cycles:
            invalid_engines.append(int(engine_id))

    passed = len(invalid_engines) == 0

    return ValidationResult(
        check_name=f"{dataset_name}: cycle sequence",
        passed=passed,
        message=(
            "All engines have continuous operating cycles."
            if passed
            else f"Invalid cycle sequence for engines: {invalid_engines}"
        ),
    )


def validate_engine_counts(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    rul_dataframe: pd.DataFrame,
) -> list[ValidationResult]:
    training_engines = train_dataframe["engine_id"].nunique()
    testing_engines = test_dataframe["engine_id"].nunique()
    rul_entries = rul_dataframe["engine_id"].nunique()

    return [
        ValidationResult(
            check_name="Training engine count",
            passed=training_engines == 100,
            message=f"Found {training_engines} training engines.",
        ),
        ValidationResult(
            check_name="Testing engine count",
            passed=testing_engines == 100,
            message=f"Found {testing_engines} testing engines.",
        ),
        ValidationResult(
            check_name="Test RUL count",
            passed=rul_entries == testing_engines,
            message=(
                f"Found {rul_entries} RUL entries for "
                f"{testing_engines} test engines."
            ),
        ),
    ]


def run_fd001_validation() -> list[ValidationResult]:
    train_dataframe, test_dataframe, rul_dataframe = load_fd001_data()

    validation_results = []

    for dataframe, dataset_name in [
        (train_dataframe, "Training data"),
        (test_dataframe, "Test data"),
    ]:
        validation_results.extend(
            [
                validate_column_count(dataframe, dataset_name),
                validate_column_names(dataframe, dataset_name),
                validate_missing_values(dataframe, dataset_name),
                validate_duplicate_rows(dataframe, dataset_name),
                validate_engine_cycle_uniqueness(
                    dataframe,
                    dataset_name,
                ),
                validate_positive_engine_ids(
                    dataframe,
                    dataset_name,
                ),
                validate_positive_cycles(
                    dataframe,
                    dataset_name,
                ),
                validate_cycle_sequence(
                    dataframe,
                    dataset_name,
                ),
            ]
        )

    validation_results.extend(
        validate_engine_counts(
            train_dataframe,
            test_dataframe,
            rul_dataframe,
        )
    )

    return validation_results


if __name__ == "__main__":
    results = run_fd001_validation()

    print("\nFD001 DATA VALIDATION REPORT")
    print("=" * 60)

    failed_checks = 0

    for result in results:
        status = "PASS" if result.passed else "FAIL"

        print(
            f"[{status}] {result.check_name}: "
            f"{result.message}"
        )

        if not result.passed:
            failed_checks += 1

    print("=" * 60)

    if failed_checks == 0:
        print("All dataset validation checks passed.")
    else:
        print(f"{failed_checks} validation checks failed.")