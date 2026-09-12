# ============================================================
# SOLAR SENSE - POLYNOMIAL REGRESSION
#
# Feature : IRRADIATION
# Target  : AC_POWER
# Degree  : 2
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. FILE PATHS
# ============================================================

# Change this to your actual SolarSense dataset path
DATASET_PATH = "D:/2nd Year/ML Solor Sense/Datasets/SolarSense_Final_Preprocessed_Data.csv"

OUTPUT_DIR = "D:/2nd Year/ML Solor Sense/Outputs/Linear_Reg/polynomial_regression"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. SELECT FEATURE AND TARGET
# ============================================================

FEATURE = "IRRADIATION"
TARGET = "AC_POWER"

required_columns = [
    FEATURE,
    TARGET
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# 4. SELECT REQUIRED DATA
# ============================================================

data = df[
    [
        FEATURE,
        TARGET
    ]
].copy()


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")
print(data.isnull().sum())

data = data.dropna()

print(
    "\nShape after removing missing values:",
    data.shape
)


# ============================================================
# 6. REMOVE INVALID VALUES
# ============================================================

data = data[
    (data[FEATURE] >= 0) &
    (data[TARGET] >= 0)
].copy()

print(
    "Shape after removing invalid values:",
    data.shape
)


# ============================================================
# 7. DEFINE X AND y
# ============================================================

X = data[[FEATURE]]

y = data[TARGET]


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain samples:", len(X_train))
print("Test samples :", len(X_test))


# ============================================================
# 9. CREATE POLYNOMIAL REGRESSION MODEL
# ============================================================

DEGREE = 2

model = Pipeline(
    steps=[
        (
            "polynomial_features",
            PolynomialFeatures(
                degree=DEGREE,
                include_bias=False
            )
        ),
        (
            "linear_regression",
            LinearRegression()
        )
    ]
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)

print(
    "\nPolynomial Regression model trained successfully."
)


# ============================================================
# 11. GET POLYNOMIAL COEFFICIENTS
# ============================================================

poly = model.named_steps[
    "polynomial_features"
]

linear_model = model.named_steps[
    "linear_regression"
]

feature_names = poly.get_feature_names_out(
    [FEATURE]
)

coefficients = linear_model.coef_

intercept = linear_model.intercept_


print("\n======================================")
print("POLYNOMIAL MODEL COEFFICIENTS")
print("======================================")

print(
    f"Intercept (b0): {intercept:.6f}"
)

for feature_name, coefficient in zip(
    feature_names,
    coefficients
):

    print(
        f"{feature_name}: {coefficient:.6f}"
    )


# ============================================================
# 12. DISPLAY REGRESSION EQUATION
# ============================================================

print("\n======================================")
print("POLYNOMIAL REGRESSION EQUATION")
print("======================================")


# For degree 2:
#
# AC_POWER = b0 + b1*IRRADIATION
#                    + b2*IRRADIATION²

if DEGREE == 2:

    b1 = coefficients[0]
    b2 = coefficients[1]

    equation = (
        f"AC_POWER = {intercept:.6f} "
        f"+ ({b1:.6f} × IRRADIATION) "
        f"+ ({b2:.6f} × IRRADIATION²)"
    )

else:

    equation = (
        f"Intercept = {intercept:.6f}"
    )

    for feature_name, coefficient in zip(
        feature_names,
        coefficients
    ):

        equation += (
            f"\n{coefficient:.6f} × "
            f"{feature_name}"
        )


print(equation)


# ============================================================
# 13. TRAIN PREDICTIONS
# ============================================================

y_train_pred = model.predict(
    X_train
)


# ============================================================
# 14. TEST PREDICTIONS
# ============================================================

y_test_pred = model.predict(
    X_test
)


# ============================================================
# 15. TRAIN METRICS
# ============================================================

train_mae = mean_absolute_error(
    y_train,
    y_train_pred
)

train_mse = mean_squared_error(
    y_train,
    y_train_pred
)

train_rmse = np.sqrt(
    train_mse
)

train_r2 = r2_score(
    y_train,
    y_train_pred
)


# ============================================================
# 16. TEST METRICS
# ============================================================

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_mse = mean_squared_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(
    test_mse
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)


# ============================================================
# 17. DISPLAY TRAIN METRICS
# ============================================================

print("\n======================================")
print("TRAINING PERFORMANCE")
print("======================================")

print(
    f"MAE  : {train_mae:.6f}"
)

print(
    f"MSE  : {train_mse:.6f}"
)

print(
    f"RMSE : {train_rmse:.6f}"
)

print(
    f"R²   : {train_r2:.6f}"
)


# ============================================================
# 18. DISPLAY TEST METRICS
# ============================================================

print("\n======================================")
print("TEST PERFORMANCE")
print("======================================")

print(
    f"MAE  : {test_mae:.6f}"
)

print(
    f"MSE  : {test_mse:.6f}"
)

print(
    f"RMSE : {test_rmse:.6f}"
)

print(
    f"R²   : {test_r2:.6f}"
)


# ============================================================
# 19. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ],

    "Train": [
        train_mae,
        train_mse,
        train_rmse,
        train_r2
    ],

    "Test": [
        test_mae,
        test_mse,
        test_rmse,
        test_r2
    ]

})


metrics_path = os.path.join(
    OUTPUT_DIR,
    "metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 20. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual_AC_POWER":
        y_test.values,

    "Predicted_AC_POWER":
        y_test_pred,

    "Error":
        y_test.values - y_test_pred

})


prediction_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 21. SAVE TRAIN PREDICTIONS
# ============================================================

train_prediction_df = pd.DataFrame({

    "Actual_AC_POWER":
        y_train.values,

    "Predicted_AC_POWER":
        y_train_pred,

    "Error":
        y_train.values - y_train_pred

})


train_prediction_path = os.path.join(
    OUTPUT_DIR,
    "train_predictions.csv"
)

train_prediction_df.to_csv(
    train_prediction_path,
    index=False
)


# ============================================================
# 22. PLOT 1
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    y_test,
    y_test_pred,
    alpha=0.6
)


# Perfect prediction line

min_value = min(
    y_test.min(),
    y_test_pred.min()
)

max_value = max(
    y_test.max(),
    y_test_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual AC_POWER"
)

plt.ylabel(
    "Predicted AC_POWER"
)

plt.title(
    "Actual vs Predicted AC_POWER\n"
    "Polynomial Regression (Degree 2)"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


actual_predicted_path = os.path.join(
    OUTPUT_DIR,
    "actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300
)

plt.show()


# ============================================================
# 23. PLOT 2
# IRRADIATION VS AC_POWER
# ============================================================

# Sort values for a smooth polynomial curve

sorted_indices = np.argsort(
    X_test[FEATURE].values.flatten()
)

X_sorted = X_test.iloc[
    sorted_indices
]

y_sorted_pred = y_test_pred[
    sorted_indices
]


plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    X_test[FEATURE],
    y_test,
    alpha=0.5,
    label="Actual"
)

plt.plot(
    X_sorted[FEATURE],
    y_sorted_pred,
    linewidth=2,
    label="Polynomial Curve"
)

plt.xlabel(
    "IRRADIATION"
)

plt.ylabel(
    "AC_POWER"
)

plt.title(
    "IRRADIATION vs AC_POWER\n"
    "Polynomial Regression (Degree 2)"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


polynomial_curve_path = os.path.join(
    OUTPUT_DIR,
    "polynomial_curve.png"
)

plt.savefig(
    polynomial_curve_path,
    dpi=300
)

plt.show()


# ============================================================
# 24. PLOT 3
# RESIDUALS VS PREDICTED
# ============================================================

residuals = (
    y_test.values -
    y_test_pred
)

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    y_test_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted AC_POWER"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Residuals vs Predicted AC_POWER\n"
    "Polynomial Regression"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


residual_path = os.path.join(
    OUTPUT_DIR,
    "residuals_vs_predicted.png"
)

plt.savefig(
    residual_path,
    dpi=300
)

plt.show()


# ============================================================
# 25. PLOT 4
# TRAIN VS TEST METRICS
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 5)
)

metric_names = [
    "MAE",
    "MSE",
    "RMSE",
    "R²"
]

train_values = [
    train_mae,
    train_mse,
    train_rmse,
    train_r2
]

test_values = [
    test_mae,
    test_mse,
    test_rmse,
    test_r2
]


for i in range(4):

    axes[i].bar(
        ["Train", "Test"],
        [
            train_values[i],
            test_values[i]
        ]
    )

    axes[i].set_title(
        metric_names[i]
    )

    axes[i].grid(
        axis="y",
        alpha=0.3
    )


plt.suptitle(
    "Polynomial Regression - "
    "Train vs Test Metrics"
)

plt.tight_layout()


metrics_plot_path = os.path.join(
    OUTPUT_DIR,
    "train_test_metrics.png"
)

plt.savefig(
    metrics_plot_path,
    dpi=300
)

plt.show()


# ============================================================
# 26. SAVE MODEL SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "model_summary.txt"
)

with open(
    summary_path,
    "w"
) as file:

    file.write(
        "SOLAR SENSE - POLYNOMIAL REGRESSION\n"
    )

    file.write(
        "====================================\n\n"
    )

    file.write(
        f"Feature : {FEATURE}\n"
    )

    file.write(
        f"Target  : {TARGET}\n"
    )

    file.write(
        f"Degree  : {DEGREE}\n\n"
    )

    file.write(
        "REGRESSION EQUATION:\n"
    )

    file.write(
        equation + "\n\n"
    )

    file.write(
        "COEFFICIENTS:\n"
    )

    file.write(
        f"Intercept: {intercept:.6f}\n"
    )

    for feature_name, coefficient in zip(
        feature_names,
        coefficients
    ):

        file.write(
            f"{feature_name}: "
            f"{coefficient:.6f}\n"
        )

    file.write(
        "\nTRAINING METRICS:\n"
    )

    file.write(
        f"MAE  : {train_mae:.6f}\n"
    )

    file.write(
        f"MSE  : {train_mse:.6f}\n"
    )

    file.write(
        f"RMSE : {train_rmse:.6f}\n"
    )

    file.write(
        f"R2   : {train_r2:.6f}\n"
    )

    file.write(
        "\nTEST METRICS:\n"
    )

    file.write(
        f"MAE  : {test_mae:.6f}\n"
    )

    file.write(
        f"MSE  : {test_mse:.6f}\n"
    )

    file.write(
        f"RMSE : {test_rmse:.6f}\n"
    )

    file.write(
        f"R2   : {test_r2:.6f}\n"
    )


# ============================================================
# 27. FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("POLYNOMIAL REGRESSION COMPLETED")
print("==========================================")

print(
    f"Metrics             : {metrics_path}"
)

print(
    f"Train Predictions   : {train_prediction_path}"
)

print(
    f"Test Predictions    : {prediction_path}"
)

print(
    f"Model Summary       : {summary_path}"
)

print(
    f"Actual vs Predicted : {actual_predicted_path}"
)

print(
    f"Polynomial Curve    : {polynomial_curve_path}"
)

print(
    f"Residual Plot       : {residual_path}"
)

print(
    f"Train/Test Metrics  : {metrics_plot_path}"
)