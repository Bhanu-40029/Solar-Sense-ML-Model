# ============================================================
# SOLARSENSE - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# 1. PATHS
# ============================================================

dataset_path = (
    "D:/2nd Year/ML Solor Sense/Datasets/"
    "SolarSense_Day_Night_Balanced_Raw.csv"
)

output_folder = (
    "D:/2nd Year/ML Solor Sense/outputs/EDA"
)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

print("=" * 70)
print("SOLARSENSE - EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 3. DATE/TIME CONVERSION
# ============================================================

df["DATE_TIME"] = pd.to_datetime(
    df["DATE_TIME"],
    dayfirst=True,
    errors="coerce"
)


# ============================================================
# 4. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC DATASET INFORMATION")
print("=" * 70)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# ============================================================
# 5. MISSING VALUES
# ============================================================

missing_values = df.isnull().sum()

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(missing_values)

missing_values.to_csv(
    os.path.join(output_folder, "missing_values.csv")
)


# ============================================================
# 6. DUPLICATE VALUES
# ============================================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows:", duplicate_count)


# ============================================================
# 7. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

summary = df.describe()

print(summary)

summary.to_csv(
    os.path.join(output_folder, "statistical_summary.csv")
)


# ============================================================
# 8. NUMERICAL COLUMNS
# ============================================================

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print("\nNumerical Columns:")
print(list(numerical_columns))


# ============================================================
# 9. DAY / NIGHT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("DAY / NIGHT DISTRIBUTION")
print("=" * 70)

period_counts = df["PERIOD"].value_counts()

print(period_counts)

plt.figure(figsize=(8, 6))

period_counts.plot(
    kind="bar"
)

plt.title("Day vs Night Record Distribution")
plt.xlabel("Period")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "day_night_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 10. AC POWER DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["AC_POWER"],
    bins=50,
    kde=True
)

plt.title("AC Power Distribution")
plt.xlabel("AC Power")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "ac_power_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 11. IRRADIATION DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["IRRADIATION"],
    bins=50,
    kde=True
)

plt.title("Irradiation Distribution")
plt.xlabel("Irradiation")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "irradiation_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 12. AMBIENT TEMPERATURE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["AMBIENT_TEMPERATURE"],
    bins=40,
    kde=True
)

plt.title("Ambient Temperature Distribution")
plt.xlabel("Ambient Temperature")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "ambient_temperature_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 13. MODULE TEMPERATURE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["MODULE_TEMPERATURE"],
    bins=40,
    kde=True
)

plt.title("Module Temperature Distribution")
plt.xlabel("Module Temperature")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "module_temperature_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 14. AC POWER OVER TIME
# ============================================================

time_data = (
    df.groupby("DATE_TIME")["AC_POWER"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(14, 6))

plt.plot(
    time_data["DATE_TIME"],
    time_data["AC_POWER"]
)

plt.title("Average AC Power Over Time")
plt.xlabel("Date and Time")
plt.ylabel("Average AC Power")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "ac_power_over_time.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 15. IRRADIATION OVER TIME
# ============================================================

irradiation_time = (
    df.groupby("DATE_TIME")["IRRADIATION"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(14, 6))

plt.plot(
    irradiation_time["DATE_TIME"],
    irradiation_time["IRRADIATION"]
)

plt.title("Average Irradiation Over Time")
plt.xlabel("Date and Time")
plt.ylabel("Irradiation")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "irradiation_over_time.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 16. CREATE HOUR FOR EDA ONLY
# ============================================================

df["HOUR"] = df["DATE_TIME"].dt.hour


# ============================================================
# 17. AVERAGE AC POWER BY HOUR
# ============================================================

hourly_power = (
    df.groupby("HOUR")["AC_POWER"]
    .mean()
)

plt.figure(figsize=(10, 6))

plt.plot(
    hourly_power.index,
    hourly_power.values,
    marker="o"
)

plt.title("Average AC Power by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average AC Power")

plt.xticks(range(0, 24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "average_ac_power_by_hour.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 18. AVERAGE IRRADIATION BY HOUR
# ============================================================

hourly_irradiation = (
    df.groupby("HOUR")["IRRADIATION"]
    .mean()
)

plt.figure(figsize=(10, 6))

plt.plot(
    hourly_irradiation.index,
    hourly_irradiation.values,
    marker="o"
)

plt.title("Average Irradiation by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Irradiation")

plt.xticks(range(0, 24))

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "average_irradiation_by_hour.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 19. IRRADIATION VS AC POWER
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="IRRADIATION",
    y="AC_POWER",
    alpha=0.3
)

plt.title("Irradiation vs AC Power")
plt.xlabel("Irradiation")
plt.ylabel("AC Power")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "irradiation_vs_ac_power.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 20. MODULE TEMPERATURE VS AC POWER
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="MODULE_TEMPERATURE",
    y="AC_POWER",
    alpha=0.3
)

plt.title("Module Temperature vs AC Power")
plt.xlabel("Module Temperature")
plt.ylabel("AC Power")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "module_temperature_vs_ac_power.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 21. AMBIENT TEMPERATURE VS AC POWER
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="AMBIENT_TEMPERATURE",
    y="AC_POWER",
    alpha=0.3
)

plt.title("Ambient Temperature vs AC Power")
plt.xlabel("Ambient Temperature")
plt.ylabel("AC Power")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "ambient_temperature_vs_ac_power.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 22. CORRELATION MATRIX
# ============================================================

correlation = df[numerical_columns].corr()

print("\n" + "=" * 70)
print("CORRELATION WITH AC_POWER")
print("=" * 70)

print(
    correlation["AC_POWER"]
    .sort_values(ascending=False)
)

correlation.to_csv(
    os.path.join(
        output_folder,
        "correlation_matrix.csv"
    )
)


# ============================================================
# 23. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "correlation_heatmap.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 24. INVERTER-WISE AVERAGE AC POWER
# ============================================================

inverter_power = (
    df.groupby("INVERTER_ID")["AC_POWER"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))

inverter_power.plot(
    kind="bar"
)

plt.title("Average AC Power by Inverter")
plt.xlabel("Inverter ID")
plt.ylabel("Average AC Power")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "inverter_average_ac_power.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 25. INVERTER-WISE MAXIMUM AC POWER
# ============================================================

inverter_max_power = (
    df.groupby("INVERTER_ID")["AC_POWER"]
    .max()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))

inverter_max_power.plot(
    kind="bar"
)

plt.title("Maximum AC Power by Inverter")
plt.xlabel("Inverter ID")
plt.ylabel("Maximum AC Power")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "inverter_max_ac_power.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 26. AC POWER BOXPLOT
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    y=df["AC_POWER"]
)

plt.title("AC Power Boxplot")
plt.ylabel("AC Power")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "ac_power_boxplot.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 27. IRRADIATION BOXPLOT
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    y=df["IRRADIATION"]
)

plt.title("Irradiation Boxplot")
plt.ylabel("Irradiation")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "irradiation_boxplot.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 28. AMBIENT TEMPERATURE BOXPLOT
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    y=df["AMBIENT_TEMPERATURE"]
)

plt.title("Ambient Temperature Boxplot")
plt.ylabel("Ambient Temperature")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "ambient_temperature_boxplot.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 29. MODULE TEMPERATURE BOXPLOT
# ============================================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    y=df["MODULE_TEMPERATURE"]
)

plt.title("Module Temperature Boxplot")
plt.ylabel("Module Temperature")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "module_temperature_boxplot.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 30. ZERO VALUE ANALYSIS
# ============================================================

zero_analysis = {}

for column in numerical_columns:

    zero_count = (df[column] == 0).sum()

    zero_percentage = (
        zero_count / len(df)
    ) * 100

    zero_analysis[column] = [
        zero_count,
        zero_percentage
    ]

zero_df = pd.DataFrame(
    zero_analysis,
    index=[
        "Zero Count",
        "Zero Percentage"
    ]
).T

print("\n" + "=" * 70)
print("ZERO VALUE ANALYSIS")
print("=" * 70)

print(zero_df)

zero_df.to_csv(
    os.path.join(
        output_folder,
        "zero_value_analysis.csv"
    )
)


# ============================================================
# 31. NEGATIVE VALUE ANALYSIS
# ============================================================

negative_analysis = {}

for column in numerical_columns:

    negative_count = (
        df[column] < 0
    ).sum()

    negative_analysis[column] = negative_count

negative_df = pd.DataFrame(
    negative_analysis.items(),
    columns=[
        "Column",
        "Negative_Value_Count"
    ]
)

print("\n" + "=" * 70)
print("NEGATIVE VALUE ANALYSIS")
print("=" * 70)

print(negative_df)

negative_df.to_csv(
    os.path.join(
        output_folder,
        "negative_value_analysis.csv"
    ),
    index=False
)


# ============================================================
# 32. FINAL EDA SUMMARY
# ============================================================

eda_summary = {
    "Total Rows": len(df),
    "Total Columns": len(df.columns),
    "Missing Values": df.isnull().sum().sum(),
    "Duplicate Rows": df.duplicated().sum(),
    "Number of Inverters": df["INVERTER_ID"].nunique(),
    "Number of Plants": df["PLANT_ID"].nunique(),
    "Number of Weather Sensors": df["WEATHER_SENSOR_ID"].nunique(),
    "Start Date": df["DATE_TIME"].min(),
    "End Date": df["DATE_TIME"].max(),
    "Average AC Power": df["AC_POWER"].mean(),
    "Maximum AC Power": df["AC_POWER"].max(),
    "Minimum AC Power": df["AC_POWER"].min(),
    "Average Irradiation": df["IRRADIATION"].mean(),
    "Maximum Irradiation": df["IRRADIATION"].max()
}

eda_summary_df = pd.DataFrame(
    eda_summary.items(),
    columns=["Metric", "Value"]
)

eda_summary_df.to_csv(
    os.path.join(
        output_folder,
        "eda_summary.csv"
    ),
    index=False
)


# ============================================================
# 33. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nAll EDA outputs saved to:")
print(output_folder)

print("\nGenerated files include:")
print("- Distribution plots")
print("- Time-series plots")
print("- Scatter plots")
print("- Correlation heatmap")
print("- Inverter analysis")
print("- Boxplots")
print("- Zero-value analysis")
print("- Negative-value analysis")
print("- Statistical summary")
print("- EDA summary")