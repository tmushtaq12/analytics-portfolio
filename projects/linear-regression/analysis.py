from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "raw" / "titanic.csv"
IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)


def load_data():
    """Load the Titanic data and keep only variables known before the trip."""
    data = pd.read_csv(DATA_PATH)

    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    columns = ["Pclass", "Sex", "Age", "FamilySize", "Embarked", "Fare"]
    return data[columns].dropna()


def build_model():
    """Build a readable preprocessing and linear-regression pipeline."""
    numeric_features = ["Pclass", "Age", "FamilySize"]
    categorical_features = ["Sex", "Embarked"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def save_actual_vs_predicted_chart(actual, predicted):
    """Save a diagnostic chart for checking prediction quality."""
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.45, color="#e76f51")
    maximum = max(actual.max(), predicted.max())
    plt.plot([0, maximum], [0, maximum], linestyle="--", color="#264653")
    plt.title("Actual vs predicted ticket fare")
    plt.xlabel("Actual fare")
    plt.ylabel("Predicted fare")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "linear_regression_actual_vs_predicted.png", dpi=160)
    plt.close()


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

    save_actual_vs_predicted_chart(y_test, predictions)

    print("Titanic Fare Linear Regression")
    print("=" * 60)
    print(f"Rows used: {len(data):,}")
    print(f"Training rows: {len(x_train):,}")
    print(f"Test rows: {len(x_test):,}")
    print("\nModel metrics:")
    print(f"MAE:  ${model_mae:,.2f}")
    print(f"RMSE: ${model_rmse:,.2f}")
    print(f"R2:   {model_r2:.3f}")
    print("\nMean-fare baseline:")
    print(f"MAE:  ${baseline_mae:,.2f}")
    print(f"RMSE: ${baseline_rmse:,.2f}")
    print("\nInterpretation:")
    if model_mae < baseline_mae:
        print("The model improves on the mean-fare baseline for this test split.")
    else:
        print("The model does not improve on the mean-fare baseline for this test split.")
    print("Chart saved: images/linear_regression_actual_vs_predicted.png")


if __name__ == "__main__":
    main()
