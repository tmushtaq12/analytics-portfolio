from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "titanic.csv"
IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = ["Pclass", "Age", "FamilySize"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def load_data() -> pd.DataFrame:
    """Load the Titanic data and create features without fitting imputers."""
    data = pd.read_csv(DATA_PATH)
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    columns = ["Pclass", "Sex", "Age", "FamilySize", "Embarked", "Fare"]
    return data[columns].dropna(subset=["Fare"])


def build_model(regressor=None) -> Pipeline:
    """Build a shared preprocessing pipeline with the requested estimator."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor or LinearRegression()),
        ]
    )


def build_log_target_model() -> TransformedTargetRegressor:
    """Fit fare in log space and return predictions in the original currency."""
    return TransformedTargetRegressor(
        regressor=build_model(),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def build_neural_network_model() -> TransformedTargetRegressor:
    """Build a small regularized MLP trained on log fares and scored in GBP."""
    return TransformedTargetRegressor(
        regressor=build_model(
            MLPRegressor(
                hidden_layer_sizes=(16,),
                solver="lbfgs",
                alpha=0.01,
                max_iter=2000,
                random_state=42,
            )
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )


def save_diagnostics(
    data: pd.DataFrame,
    actual: pd.Series,
    predicted: pd.Series,
    model: Pipeline,
    age_test: pd.DataFrame,
    age_predictions: np.ndarray,
    comparison: pd.DataFrame,
) -> None:
    """Save best-fit line, residual, coefficient, model comparison, and fare charts."""
    residuals = actual - predicted
    age_order = np.argsort(age_test["Age"].to_numpy())
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["regressor"].coef_
    coefficient_table = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    coefficient_table["feature"] = (
        coefficient_table["feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    coefficient_table["absolute_value"] = coefficient_table["coefficient"].abs()
    coefficient_table = coefficient_table.sort_values("absolute_value", ascending=False).head(10)

    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    figure.suptitle("Titanic Fare Model Comparison and Diagnostics", fontsize=18, fontweight="bold")

    maximum = max(actual.max(), predicted.max())
    axes[0, 0].scatter(age_test["Age"], actual, alpha=0.4, color="#2a9d8f", edgecolors="white", linewidth=0.3)
    axes[0, 0].plot(
        age_test["Age"].to_numpy()[age_order],
        age_predictions[age_order],
        color="#000080",
        linewidth=2,
    )
    axes[0, 0].set_title("One-feature line: age and fare")
    axes[0, 0].set_xlabel("Age in years")
    axes[0, 0].set_ylabel("Fare in GBP")

    axes[0, 1].scatter(actual, predicted, alpha=0.48, color="#e76f51", edgecolors="white", linewidth=0.3)
    axes[0, 1].plot([0, maximum], [0, maximum], linestyle="--", color="#264653", linewidth=1.5)
    axes[0, 1].set_title("Multiple linear model: actual vs predicted")
    axes[0, 1].set_xlabel("Actual fare")
    axes[0, 1].set_ylabel("Predicted fare")
    axes[0, 1].text(0.05, 0.92, f"R2 = {r2_score(actual, predicted):.3f}", transform=axes[0, 1].transAxes)

    axes[0, 2].scatter(predicted, residuals, alpha=0.48, color="#2a9d8f", edgecolors="white", linewidth=0.3)
    axes[0, 2].axhline(0, linestyle="--", color="#264653", linewidth=1.5)
    axes[0, 2].set_title("Residuals: actual minus predicted")
    axes[0, 2].set_xlabel("Predicted fare")
    axes[0, 2].set_ylabel("Residual in GBP")

    coefficient_table = coefficient_table.sort_values("coefficient")
    axes[1, 0].barh(coefficient_table["feature"], coefficient_table["coefficient"], color="#457b9d")
    axes[1, 0].axvline(0, color="#264653", linewidth=1)
    axes[1, 0].set_title("Largest standardized linear coefficients")
    axes[1, 0].set_xlabel("Coefficient direction and size")

    sns.boxplot(data=data, x="Pclass", y="Fare", hue="Pclass", legend=False, palette="Set2", ax=axes[1, 1])
    axes[1, 1].set_title("Observed fare distribution by class")
    axes[1, 1].set_xlabel("Passenger class")
    axes[1, 1].set_ylabel("Fare")
    axes[1, 1].set_ylim(0, data["Fare"].quantile(0.95))

    comparison_plot = comparison.sort_values("test_mae", ascending=True)
    axes[1, 2].barh(comparison_plot["model"], comparison_plot["test_mae"], color="#e09f3e")
    axes[1, 2].set_title("Held-out test MAE (lower is better)")
    axes[1, 2].set_xlabel("Mean absolute error in GBP")

    figure.tight_layout()
    figure.savefig(IMAGE_DIR / "linear_regression_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    coefficient_table.to_csv(BASE_DIR / "projects" / "linear-regression" / "coefficient_summary.csv", index=False)
    comparison.to_csv(BASE_DIR / "projects" / "linear-regression" / "model_comparison.csv", index=False)


def main():
    data = load_data()
    features = data.drop(columns="Fare")
    target = data["Fare"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    age_fill_value = x_train["Age"].median()
    age_train = x_train[["Age"]].fillna(age_fill_value)
    age_test = x_test[["Age"]].fillna(age_fill_value)
    age_line = LinearRegression().fit(age_train, y_train)
    age_predictions = age_line.predict(age_test)

    models = {
        "multiple_linear": build_model(),
        "log_target_linear": build_log_target_model(),
        "neural_network_mlp_log_target": build_neural_network_model(),
    }
    fitted_models = {}
    rows = [{"model": "mean_baseline", "test_mae": None, "test_rmse": None, "test_r2": None}]
    baseline_predictions = np.full(len(y_test), y_train.mean())
    rows[0]["test_mae"] = mean_absolute_error(y_test, baseline_predictions)
    rows[0]["test_rmse"] = mean_squared_error(y_test, baseline_predictions) ** 0.5
    rows[0]["test_r2"] = r2_score(y_test, baseline_predictions)

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        fitted_models[model_name] = model
        predictions = model.predict(x_test)
        cross_validation_model = {
            "multiple_linear": build_model,
            "log_target_linear": build_log_target_model,
            "neural_network_mlp_log_target": build_neural_network_model,
        }[model_name]()
        cross_validation = cross_validate(
            cross_validation_model,
            features,
            target,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        )
        rows.append(
            {
                "model": model_name,
                "test_mae": mean_absolute_error(y_test, predictions),
                "test_rmse": mean_squared_error(y_test, predictions) ** 0.5,
                "test_r2": r2_score(y_test, predictions),
                "cv_mae_mean": -cross_validation["test_mae"].mean(),
                "cv_mae_std": cross_validation["test_mae"].std(),
                "cv_r2_mean": cross_validation["test_r2"].mean(),
                "cv_r2_std": cross_validation["test_r2"].std(),
            }
        )

    comparison = pd.DataFrame(rows)
    linear_model = fitted_models["multiple_linear"]
    linear_predictions = linear_model.predict(x_test)
    save_diagnostics(
        data,
        y_test,
        linear_predictions,
        linear_model,
        age_test,
        age_predictions,
        comparison,
    )

    print("Titanic Fare Regression Model Comparison")
    print("=" * 60)
    print(f"Rows used: {len(data):,}")
    print(f"Training rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")
    print("\nTest metrics and cross-validation:")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nInterpretation: compare model scores with the baseline and inspect errors; the lowest test MAE is not proof of a generally superior model.")
    print("Charts saved: images/linear_regression_diagnostics.png")
    print("Coefficients saved: projects/linear-regression/coefficient_summary.csv")
    print("Model comparison saved: projects/linear-regression/model_comparison.csv")


if __name__ == "__main__":
    main()
