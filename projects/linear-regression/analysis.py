from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "titanic.csv"
IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = ["Pclass", "Age", "FamilySize"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def load_data() -> pd.DataFrame:
    """Load the Titanic data and keep only variables known before the trip."""
    data = pd.read_csv(DATA_PATH)

    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    columns = ["Pclass", "Sex", "Age", "FamilySize", "Embarked", "Fare"]
    return data[columns].dropna()


def build_model() -> Pipeline:
    """Build a readable preprocessing and linear-regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def save_diagnostics(data: pd.DataFrame, actual, predicted, model: Pipeline) -> None:
    """Save model fit, residual, coefficient, and distribution visuals."""
    residuals = actual - predicted
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
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.suptitle("Titanic Fare Regression Diagnostics", fontsize=18, fontweight="bold")

    maximum = max(actual.max(), predicted.max())
    axes[0, 0].scatter(actual, predicted, alpha=0.48, color="#e76f51", edgecolors="white", linewidth=0.3)
    axes[0, 0].plot([0, maximum], [0, maximum], linestyle="--", color="#264653", linewidth=1.5)
    axes[0, 0].set_title("Predictions track broad fare levels")
    axes[0, 0].set_xlabel("Actual fare")
    axes[0, 0].set_ylabel("Predicted fare")
    axes[0, 0].text(0.05, 0.92, f"R² = {r2_score(actual, predicted):.3f}", transform=axes[0, 0].transAxes)

    axes[0, 1].scatter(predicted, residuals, alpha=0.48, color="#2a9d8f", edgecolors="white", linewidth=0.3)
    axes[0, 1].axhline(0, linestyle="--", color="#264653", linewidth=1.5)
    axes[0, 1].set_title("Residuals reveal where the model misses")
    axes[0, 1].set_xlabel("Predicted fare")
    axes[0, 1].set_ylabel("Actual - predicted")

    coefficient_table = coefficient_table.sort_values("coefficient")
    axes[1, 0].barh(coefficient_table["feature"], coefficient_table["coefficient"], color="#457b9d")
    axes[1, 0].axvline(0, color="#264653", linewidth=1)
    axes[1, 0].set_title("Largest standardized model coefficients")
    axes[1, 0].set_xlabel("Coefficient direction and size")

    sns.boxplot(data=data, x="Pclass", y="Fare", hue="Pclass", legend=False, palette="Set2", ax=axes[1, 1])
    axes[1, 1].set_title("Observed fare distribution by class")
    axes[1, 1].set_xlabel("Passenger class")
    axes[1, 1].set_ylabel("Fare")
    axes[1, 1].set_ylim(0, data["Fare"].quantile(0.95))

    figure.tight_layout()
    figure.savefig(IMAGE_DIR / "linear_regression_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    coefficient_table.to_csv(BASE_DIR / "projects" / "linear-regression" / "coefficient_summary.csv", index=False)


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

    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    baseline_predictions = [y_train.mean()] * len(y_test)

    model_mae = mean_absolute_error(y_test, predictions)
    model_rmse = mean_squared_error(y_test, predictions) ** 0.5
    model_r2 = r2_score(y_test, predictions)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_rmse = mean_squared_error(y_test, baseline_predictions) ** 0.5

    cross_validation = cross_validate(
        build_model(),
        features,
        target,
        cv=5,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
    )
    cv_mae = -cross_validation["test_mae"]
    cv_r2 = cross_validation["test_r2"]

    save_diagnostics(data, y_test, predictions, model)

    print("Titanic Fare Linear Regression")
    print("=" * 60)
    print(f"Rows used: {len(data):,}")
    print(f"Training rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")
    print("\nModel metrics:")
    print(f"MAE:  ${model_mae:,.2f}")
    print(f"RMSE: ${model_rmse:,.2f}")
    print(f"R2:   {model_r2:.3f}")
    print("\n5-fold cross-validation:")
    print(f"MAE:  ${cv_mae.mean():,.2f} +/- ${cv_mae.std():,.2f}")
    print(f"R2:   {cv_r2.mean():.3f} +/- {cv_r2.std():.3f}")
    print("\nMean-fare baseline:")
    print(f"MAE:  ${baseline_mae:,.2f}")
    print(f"RMSE: ${baseline_rmse:,.2f}")
    print("\nInterpretation:")
    if model_mae < baseline_mae:
        print("The model improves on the mean-fare baseline for this test split.")
    else:
        print("The model does not improve on the mean-fare baseline for this test split.")
    print("Charts saved: images/linear_regression_diagnostics.png")
    print("Coefficients saved: projects/linear-regression/coefficient_summary.csv")


if __name__ == "__main__":
    main()
