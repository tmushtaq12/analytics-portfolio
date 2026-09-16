# Linear Regression: Titanic Fare Prediction

## Objective
Build a simple, interpretable machine-learning model that predicts a passenger's ticket fare from information available before the voyage.

This is a deliberately focused project. It shows the complete modeling workflow without hiding the evaluation behind a large framework.

## Target
`Fare`, the passenger's ticket price.

## Features
- `Pclass` — passenger class
- `Sex` — passenger sex
- `Age` — passenger age
- `FamilySize` — `SibSp + Parch + 1`
- `Embarked` — boarding port

The model does not use `Survived`, `Name`, `Ticket`, or `Cabin`. Those fields are either unrelated to the pricing question, too sparse, or unsuitable for this simple baseline.

## Method
1. Load the real Titanic CSV used by the Kaggle project
2. Fill missing age and embarkation values
3. Create the family-size feature
4. Split the data into 80% training and 20% test records
5. Standardize numeric features
6. One-hot encode categorical features
7. Fit `LinearRegression`
8. Compare the model against a baseline that always predicts the training-set mean fare
9. Use 5-fold cross-validation to check whether results are stable
10. Inspect residuals, coefficients, and fare distributions

All preprocessing is inside a scikit-learn `Pipeline`, so the test set is not used when fitting transformations.

## Evaluation Metrics
- **MAE**: average absolute prediction error in currency units
- **RMSE**: penalizes larger errors more heavily
- **R2**: proportion of target variation explained by the model

## Results
Run the script to reproduce the current metrics:

```powershell
python projects/linear-regression/analysis.py
```

The script prints model metrics and mean-baseline metrics, then reports whether the model improves on the baseline for the fixed test split.

The verified run produced:

| Measure | Test split | 5-fold cross-validation |
| --- | ---: | ---: |
| MAE | $20.65 | $20.58 +/- $1.82 |
| RMSE | $30.31 | Not used for CV summary |
| R2 | 0.406 | 0.385 +/- 0.060 |

The mean-fare baseline had a test MAE of `$25.69`, so the model improved the baseline while still leaving substantial unexplained variation.

## Output
![Regression diagnostic dashboard](../../images/linear_regression_diagnostics.png)

The dashboard combines four useful views:

- actual versus predicted fare, with the diagonal representing perfect predictions
- residuals versus predictions, showing where errors are concentrated
- the largest standardized coefficients, showing model direction and relative strength
- observed fare distributions by passenger class, grounding the model in the original data

The largest coefficient signal is passenger class, which is consistent with the observed fare distributions. Coefficients should be interpreted as associations in this historical dataset, not causal effects.

## Honest Limitation
This is an educational regression example, not a production pricing system. The Titanic data has a historical and unusual pricing context. The cross-validation results are more useful than a single split, but a stronger version would compare regularized regression, investigate fare outliers, and test whether ticket-level grouping changes the result.

## Files
- `analysis.py` — readable end-to-end modeling script
- `coefficient_summary.csv` — generated coefficient table for interpretation
- `../../data/raw/titanic.csv` — real public dataset used as input
- `../../images/linear_regression_diagnostics.png` — generated four-panel diagnostic dashboard
