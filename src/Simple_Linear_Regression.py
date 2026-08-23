# ============================================================
# SOLAR SENSE - SIMPLE LINEAR REGRESSION
# Feature  : IRRADIATION
# Target   : AC_POWER
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

# Change this to your actual dataset path
DATASET_PATH = "D:/2nd Year/ML Solor Sense/Datasets/SolarSense_Final_Preprocessed_Data.csv"

# All outputs will be stored here
OUTPUT_DIR = "/Outputs/Linear_Reg/simple_linear_regression"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. SELECT FEATURE AND TARGET
# ============================================================

FEATURE = "IRRADIATION"
TARGET = "AC_POWER"

required_columns = [FEATURE, TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

data = df[required_columns].copy()

print("\nMissing values before cleaning:")
print(data.isnull().sum())

data = data.dropna()

print("\nShape after removing missing values:", data.shape)


# ============================================================
# 5. REMOVE INVALID VALUES
# ============================================================

# Solar irradiation and AC power should not be negative
data = data[
    (data[FEATURE] >= 0) &
    (data[TARGET] >= 0)
].copy()

print("Shape after removing invalid values:", data.shape)


# ============================================================
# 6. DEFINE X AND y
# ============================================================

X = data[[FEATURE]]
y = data[TARGET]

print("\nInput feature:")
print(FEATURE)

print("\nTarget:")
print(TARGET)


# ============================================================
# 7. TRAIN-TEST SPLIT
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
# 8. CREATE AND TRAIN MODEL
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully.")


# ============================================================
# 9. GET MODEL PARAMETERS
# ============================================================

intercept = model.intercept_
coefficient = model.coef_[0]

print("\nLearned Regression Equation:")
print(
    f"AC_POWER = {intercept:.4f} + "
    f"({coefficient:.4f} × IRRADIATION)"
)


# ============================================================
# 10. PREDICTIONS
# ============================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# ============================================================
# 11. CALCULATE TRAINING METRICS
# ============================================================

train_mae = mean_absolute_error(
    y_train,
    y_train_pred
)

train_mse = mean_squared_error(
    y_train,
    y_train_pred
)

train_rmse = np.sqrt(train_mse)

train_r2 = r2_score(
    y_train,
    y_train_pred
)


# ============================================================
# 12. CALCULATE TEST METRICS
# ============================================================

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_mse = mean_squared_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(test_mse)

test_r2 = r2_score(
    y_test,
    y_test_pred
)


# ============================================================
# 13. DISPLAY METRICS
# ============================================================

print("\n==============================")
print("TRAINING PERFORMANCE")
print("==============================")

print(f"MAE  : {train_mae:.4f}")
print(f"MSE  : {train_mse:.4f}")
print(f"RMSE : {train_rmse:.4f}")
print(f"R²   : {train_r2:.4f}")


print("\n==============================")
print("TEST PERFORMANCE")
print("==============================")

print(f"MAE  : {test_mae:.4f}")
print(f"MSE  : {test_mse:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"R²   : {test_r2:.4f}")


# ============================================================
# 14. SAVE METRICS TO CSV
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
# 15. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "Actual_AC_POWER": y_test.values,
    "Predicted_AC_POWER": y_test_pred,
    "Error": y_test.values - y_test_pred
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
# 16. PLOT 1
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 6))

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

plt.xlabel("Actual AC_POWER")
plt.ylabel("Predicted AC_POWER")

plt.title(
    "Actual vs Predicted AC_POWER\n"
    "Simple Linear Regression"
)

plt.grid(True, alpha=0.3)

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
# 17. PLOT 2
# IRRADIATION VS AC_POWER + REGRESSION LINE
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    X_test[FEATURE],
    y_test,
    alpha=0.5,
    label="Actual"
)

# Sort X for a clean regression line
sorted_indices = np.argsort(
    X_test[FEATURE].values.flatten()
)

X_sorted = X_test.iloc[
    sorted_indices
]

y_sorted_pred = y_test_pred[
    sorted_indices
]

plt.plot(
    X_sorted[FEATURE],
    y_sorted_pred,
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("IRRADIATION")
plt.ylabel("AC_POWER")

plt.title(
    "IRRADIATION vs AC_POWER\n"
    "Simple Linear Regression"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

regression_line_path = os.path.join(
    OUTPUT_DIR,
    "regression_line.png"
)

plt.savefig(
    regression_line_path,
    dpi=300
)

plt.show()


# ============================================================
# 18. PLOT 3
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
    "Simple Linear Regression - Train vs Test Metrics"
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
# 19. SAVE MODEL SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "model_summary.txt"
)

with open(summary_path, "w") as file:

    file.write(
        "SOLAR SENSE - SIMPLE LINEAR REGRESSION\n"
    )

    file.write(
        "=======================================\n\n"
    )

    file.write(
        f"Feature : {FEATURE}\n"
    )

    file.write(
        f"Target  : {TARGET}\n\n"
    )

    file.write(
        "Regression Equation:\n"
    )

    file.write(
        f"AC_POWER = {intercept:.6f} + "
        f"({coefficient:.6f} × IRRADIATION)\n\n"
    )

    file.write(
        "TRAINING METRICS\n"
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
        f"R2   : {train_r2:.6f}\n\n"
    )

    file.write(
        "TEST METRICS\n"
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
# 20. FINAL OUTPUT LOCATIONS
# ============================================================

print("\n======================================")
print("FILES SAVED SUCCESSFULLY")
print("======================================")

print(f"Metrics       : {metrics_path}")
print(f"Predictions   : {prediction_path}")
print(f"Model Summary : {summary_path}")
print(f"Plot 1        : {actual_predicted_path}")
print(f"Plot 2        : {regression_line_path}")
print(f"Plot 3        : {metrics_plot_path}")