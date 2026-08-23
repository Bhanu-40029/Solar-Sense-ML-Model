# ============================================================
# SOLAR SENSE - MULTIPLE LINEAR REGRESSION
#
# Features:
#   IRRADIATION
#   AMBIENT_TEMPERATURE
#   MODULE_TEMPERATURE
#   HOUR
#
# Target:
#   AC_POWER
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. FILE PATHS
# ============================================================

# CHANGE THIS TO YOUR ACTUAL DATASET PATH
DATASET_PATH = "D:/2nd Year/ML Solor Sense/Datasets/SolarSense_Final_Preprocessed_Data.csv"

# Output folder
OUTPUT_DIR = "D:/2nd Year/ML Solor Sense/Outputs/Linear_Reg/multiple_linear_regression"

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
# 3. SELECT FEATURES AND TARGET
# ============================================================

FEATURES = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "HOUR"
]

TARGET = "AC_POWER"


# Check whether required columns exist

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"These columns are missing from the dataset: "
        f"{missing_columns}"
    )


# ============================================================
# 4. CREATE DATAFRAME WITH REQUIRED COLUMNS
# ============================================================

data = df[required_columns].copy()

print("\nSelected columns:")
print(data.columns.tolist())


# ============================================================
# 5. CHECK MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")
print(data.isnull().sum())


# Remove rows containing missing values

data = data.dropna()

print(
    "\nDataset shape after removing missing values:",
    data.shape
)


# ============================================================
# 6. REMOVE INVALID VALUES
# ============================================================

# Solar irradiation and AC power should not be negative

data = data[
    (data["IRRADIATION"] >= 0) &
    (data["AC_POWER"] >= 0)
].copy()

print(
    "Dataset shape after removing invalid values:",
    data.shape
)


# ============================================================
# 7. DEFINE INPUT X AND TARGET y
# ============================================================

X = data[FEATURES]

y = data[TARGET]

print("\nInput features:")
print(FEATURES)

print("\nTarget:")
print(TARGET)


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
# 9. CREATE MULTIPLE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 10. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)

print("\nMultiple Linear Regression model trained successfully.")


# ============================================================
# 11. GET MODEL COEFFICIENTS
# ============================================================

intercept = model.intercept_

coefficients = model.coef_


print("\n======================================")
print("LEARNED MODEL COEFFICIENTS")
print("======================================")

print(
    f"Intercept (b0): {intercept:.6f}"
)

for feature, coefficient in zip(
    FEATURES,
    coefficients
):

    print(
        f"{feature}: {coefficient:.6f}"
    )


# ============================================================
# 12. DISPLAY REGRESSION EQUATION
# ============================================================

print("\n======================================")
print("REGRESSION EQUATION")
print("======================================")

equation = f"AC_POWER = {intercept:.6f}"

for feature, coefficient in zip(
    FEATURES,
    coefficients
):

    if coefficient >= 0:

        equation += (
            f" + ({coefficient:.6f} × {feature})"
        )

    else:

        equation += (
            f" - ({abs(coefficient):.6f} × {feature})"
        )

print(equation)


# ============================================================
# 13. PREDICT TRAINING DATA
# ============================================================

y_train_pred = model.predict(
    X_train
)


# ============================================================
# 14. PREDICT TEST DATA
# ============================================================

y_test_pred = model.predict(
    X_test
)


# ============================================================
# 15. TRAINING METRICS
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

print(f"MAE  : {train_mae:.6f}")
print(f"MSE  : {train_mse:.6f}")
print(f"RMSE : {train_rmse:.6f}")
print(f"R²   : {train_r2:.6f}")


# ============================================================
# 18. DISPLAY TEST METRICS
# ============================================================

print("\n======================================")
print("TEST PERFORMANCE")
print("======================================")

print(f"MAE  : {test_mae:.6f}")
print(f"MSE  : {test_mse:.6f}")
print(f"RMSE : {test_rmse:.6f}")
print(f"R²   : {test_r2:.6f}")


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
    "Multiple Linear Regression"
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
# RESIDUALS VS PREDICTED VALUES
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
    "Multiple Linear Regression"
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
# 24. PLOT 3
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
    "Multiple Linear Regression - "
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
# 25. FEATURE COEFFICIENT PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    FEATURES,
    coefficients
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Coefficient Value"
)

plt.title(
    "Multiple Linear Regression Coefficients"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


coefficient_plot_path = os.path.join(
    OUTPUT_DIR,
    "feature_coefficients.png"
)

plt.savefig(
    coefficient_plot_path,
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
        "SOLAR SENSE - MULTIPLE LINEAR REGRESSION\n"
    )

    file.write(
        "=========================================\n\n"
    )

    file.write(
        "FEATURES:\n"
    )

    for feature in FEATURES:

        file.write(
            f"- {feature}\n"
        )

    file.write(
        f"\nTARGET:\n- {TARGET}\n\n"
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

    for feature, coefficient in zip(
        FEATURES,
        coefficients
    ):

        file.write(
            f"{feature}: {coefficient:.6f}\n"
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
print("MULTIPLE LINEAR REGRESSION COMPLETED")
print("==========================================")

print(
    f"Metrics              : {metrics_path}"
)

print(
    f"Test Predictions     : {prediction_path}"
)

print(
    f"Train Predictions    : {train_prediction_path}"
)

print(
    f"Model Summary        : {summary_path}"
)

print(
    f"Actual vs Predicted  : {actual_predicted_path}"
)

print(
    f"Residual Plot        : {residual_path}"
)

print(
    f"Train/Test Metrics   : {metrics_plot_path}"
)

print(
    f"Coefficient Plot     : {coefficient_plot_path}"
)