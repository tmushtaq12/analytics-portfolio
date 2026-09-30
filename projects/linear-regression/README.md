# Regression Models: Titanic Fare Prediction

## Objective
Compare a mean baseline, a one-feature line of best fit, a multiple linear model, a log-target linear model, and a small neural network to predict a passenger's ticket fare from information available before the voyage.

This educational comparison uses a real public dataset and a readable scikit-learn pipeline. All model variants use the same training/test split and the same preprocessor; cross-validation uses the same five shuffled folds for the three fitted models.

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
1. Load the real Titanic CSV and create family size.
2. Split the data into 80% training and 20% test records with `random_state=42`.
3. Fit numeric median imputation and standardization, plus categorical mode imputation and one-hot encoding, inside each model pipeline.
4. Fit a one-feature age-to-fare line, a mean-fare baseline, multiple linear regression, log-target linear regression, and a log-target MLP with one 16-unit hidden layer.
5. Use the same five shuffled folds (`random_state=42`) to estimate variation for each fitted model.
6. Compare MAE, RMSE, and R2 on the untouched test split and inspect residuals, a fitted age/fare line, linear coefficients, and fare distributions.

All learned preprocessing is inside scikit-learn pipelines, so each validation fold and the test split fit their own imputers and scalers from training data only. The mean baseline uses the training target mean.

## Evaluation Metrics
- **MAE**: average absolute prediction error in currency units
- **MSE**: average squared residual; emphasizes large misses and has squared currency units
- **RMSE**: square root of MSE; emphasizes large misses but returns to currency units
- **R2**: compares the model's squared residuals with a constant prediction at the evaluated target mean; it may be negative when worse than that reference
- **Residual**: `actual fare - predicted fare`; a residual plot can expose curvature, changing spread, and outliers

## Results
Run the script to reproduce the current metrics:

```powershell
python projects/linear-regression/analysis.py
```

The script prints test-split metrics and cross-validation mean/standard deviation for each fitted model.

The verified run produced the following fixed-split test scores and shuffled five-fold averages:

| Model | Test MAE | Test RMSE | Test R2 | CV MAE mean +/- SD | CV R2 mean +/- SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training-mean baseline | GBP 25.69 | GBP 39.38 | -0.002 | Not applicable | Not applicable |
| Multiple linear | GBP 20.65 | GBP 30.31 | 0.406 | GBP 20.59 +/- 2.36 | 0.422 +/- 0.098 |
| Log-target linear | GBP 10.79 | GBP 25.08 | 0.594 | GBP 13.38 +/- 3.47 | 0.493 +/- 0.153 |
| Log-target MLP neural network | GBP 12.61 | GBP 28.48 | 0.476 | GBP 12.52 +/- 2.52 | 0.529 +/- 0.154 |

The test split favors the log-target linear model on MAE, while cross-validation gives the MLP a slightly lower average MAE. That disagreement is a reason to investigate split sensitivity and uncertainty, not to declare a winner. Every listed model improves on the mean baseline in this experiment, but the historical data and chosen split limit generalization claims.

## Output
![Regression diagnostic dashboard](../../images/linear_regression_diagnostics.png)

The dashboard combines six views:

- a one-feature line of best fit for age versus fare
- multiple-linear-model actual versus predicted fare, with the diagonal representing perfect predictions
- residuals versus predictions, showing where errors are concentrated
- the largest standardized coefficients, showing model direction and relative strength
- observed fare distributions by passenger class, grounding the model in the original data
- held-out MAE by model, with lower values indicating smaller average absolute misses

The largest coefficient signal is passenger class, which is consistent with the observed fare distributions. Coefficients should be interpreted as associations in this historical dataset, not causal effects.

## Honest Limitation
This is an educational regression example, not a production pricing system. Fare is strongly skewed, and inverse-transforming a log prediction does not make the model optimize MAE on the original scale. The MLP is small and uses regularization, but it still has more flexibility than the linear models. Results are sensitive to the held-out split; tuning or model selection must happen within training data, with the final test set reserved for one final evaluation. Ticket groups may cross random splits, so a group-aware split is a useful next robustness check. No causal claim is made about the fare features.

## Files
- `analysis.py` — readable end-to-end modeling script
- `coefficient_summary.csv` — generated coefficient table for interpretation
- `model_comparison.csv` — generated held-out and cross-validation metrics for each model
- `../../data/raw/titanic.csv` — real public dataset used as input
- `../../images/linear_regression_diagnostics.png` — generated six-panel model and diagnostic dashboard
