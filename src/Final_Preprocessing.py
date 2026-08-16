# ============================================================
# SOLARSENSE - COMPLETE DATA PREPROCESSING
# ============================================================

import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. PATHS
# ============================================================

input_path = (
    "D:/2nd Year/ML Solor Sense/Datasets/"
    "SolarSense_Cleaned_Data.csv"
)

output_folder = (
    "D:/2nd Year/ML Solor Sense/Datasets"
)

os.makedirs(output_folder, exist_ok=True)

final_output_path = (
    "D:/2nd Year/ML Solor Sense/Datasets/"
    "SolarSense_Final_Preprocessed_Data.csv"
)


# ============================================================
# 2. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv(input_path)

print("=" * 70)
print("SOLARSENSE - COMPLETE DATA PREPROCESSING")
print("=" * 70)

print("\nOriginal Dataset Shape:")
print(df.shape)


# ============================================================
# 3. PRINT ORIGINAL DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL DATA TYPES")
print("=" * 70)

datatype_table = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

print(
    datatype_table.to_string(index=False)
)


# ============================================================
# 4. NUMERICAL STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("NUMERICAL STANDARDIZATION")
print("=" * 70)


# Genuine numerical measurement columns
numerical_columns = [
    "DC_POWER",
    "DAILY_YIELD",
    "TOTAL_YIELD",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "AC_POWER"
]


print("\nNumerical columns selected:")

for column in numerical_columns:
    print("-", column)


# ------------------------------------------------------------
# Check that all columns exist
# ------------------------------------------------------------

missing_numerical_columns = [
    column
    for column in numerical_columns
    if column not in df.columns
]

if missing_numerical_columns:

    print(
        "\nERROR: These numerical columns are missing:"
    )

    print(missing_numerical_columns)

    raise ValueError(
        "Required numerical columns are missing."
    )


# ------------------------------------------------------------
# Display before standardization
# ------------------------------------------------------------

print("\nBEFORE STANDARDIZATION:")

print(
    df[numerical_columns].describe()
)


# ------------------------------------------------------------
# Create scaler
# ------------------------------------------------------------

scaler = StandardScaler()


# ------------------------------------------------------------
# Standardize numerical columns
# ------------------------------------------------------------

df[numerical_columns] = scaler.fit_transform(
    df[numerical_columns]
)


# ------------------------------------------------------------
# Display after standardization
# ------------------------------------------------------------

print("\nAFTER STANDARDIZATION:")

print(
    df[numerical_columns].describe()
)


# ------------------------------------------------------------
# Check mean
# ------------------------------------------------------------

print("\nMean after standardization:")

print(
    df[numerical_columns].mean()
)


# ------------------------------------------------------------
# Check standard deviation
# ------------------------------------------------------------

print("\nStandard deviation after standardization:")

print(
    df[numerical_columns].std()
)


# ============================================================
# 5. CATEGORICAL PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL PREPROCESSING")
print("=" * 70)


# ============================================================
# 6. CHECK CATEGORICAL VALUES
# ============================================================

print("\nINVERTER_ID values:")

print(
    df["INVERTER_ID"].unique()
)


print("\nPERIOD values:")

print(
    df["PERIOD"].unique()
)


print("\nWEATHER_SENSOR_ID values:")

print(
    df["WEATHER_SENSOR_ID"].unique()
)


# ============================================================
# 7. ONE-HOT ENCODING - INVERTER_ID
# ============================================================

print("\n" + "=" * 70)
print("ONE-HOT ENCODING - INVERTER_ID")
print("=" * 70)


encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)


inverter_encoded = encoder.fit_transform(
    df[["INVERTER_ID"]]
)


inverter_columns = encoder.get_feature_names_out(
    ["INVERTER_ID"]
)


inverter_encoded_df = pd.DataFrame(
    inverter_encoded,
    columns=inverter_columns,
    index=df.index
)


print("\nGenerated inverter columns:")

for column in inverter_columns:

    print("-", column)


# ============================================================
# 8. BINARY ENCODING - PERIOD
# ============================================================

print("\n" + "=" * 70)
print("BINARY ENCODING - PERIOD")
print("=" * 70)


# Check unexpected PERIOD values
valid_periods = {
    "DAY",
    "NIGHT"
}

existing_periods = set(
    df["PERIOD"].dropna().unique()
)

unexpected_periods = (
    existing_periods - valid_periods
)


if unexpected_periods:

    print(
        "\nERROR: Unexpected PERIOD values found:"
    )

    print(unexpected_periods)

    raise ValueError(
        "Unexpected values found in PERIOD column."
    )


# DAY = 1
# NIGHT = 0

df["PERIOD"] = df["PERIOD"].map({
    "DAY": 1,
    "NIGHT": 0
})


print("\nPERIOD after encoding:")

print(
    df["PERIOD"].value_counts()
)


# ============================================================
# 9. REMOVE ORIGINAL INVERTER_ID
# ============================================================

df.drop(
    columns=["INVERTER_ID"],
    inplace=True
)


# ============================================================
# 10. ADD ONE-HOT ENCODED INVERTER COLUMNS
# ============================================================

df = pd.concat(
    [
        df,
        inverter_encoded_df
    ],
    axis=1
)


# ============================================================
# 11. DATE-TIME FEATURE EXTRACTION
# ============================================================

print("\n" + "=" * 70)
print("DATE-TIME FEATURE EXTRACTION")
print("=" * 70)


# ------------------------------------------------------------
# Convert DATE_TIME
# ------------------------------------------------------------

df["DATE_TIME"] = pd.to_datetime(
    df["DATE_TIME"],
    dayfirst=True,
    errors="coerce"
)


print("\nDATE_TIME datatype:")

print(
    df["DATE_TIME"].dtype
)


# ============================================================
# 12. CHECK INVALID DATE_TIME VALUES
# ============================================================

invalid_datetime = (
    df["DATE_TIME"].isnull().sum()
)


print("\nInvalid DATE_TIME values:")

print(
    invalid_datetime
)


# If invalid dates exist, stop instead of silently creating
# missing time features.

if invalid_datetime > 0:

    raise ValueError(
        f"{invalid_datetime} invalid DATE_TIME values found."
    )


# ============================================================
# 13. EXTRACT HOUR
# ============================================================

df["HOUR"] = (
    df["DATE_TIME"].dt.hour
)


# ============================================================
# 14. EXTRACT DAY
# ============================================================

df["DAY"] = (
    df["DATE_TIME"].dt.day
)


# ============================================================
# 15. EXTRACT MONTH
# ============================================================

df["MONTH"] = (
    df["DATE_TIME"].dt.month
)


# ============================================================
# 16. EXTRACT DAY OF WEEK
# ============================================================

# Monday = 0
# Tuesday = 1
# Wednesday = 2
# Thursday = 3
# Friday = 4
# Saturday = 5
# Sunday = 6

df["DAY_OF_WEEK"] = (
    df["DATE_TIME"].dt.dayofweek
)


# ============================================================
# 17. DISPLAY DATE-TIME FEATURES
# ============================================================

print("\nExtracted Date-Time Features:")

print(
    df[
        [
            "DATE_TIME",
            "HOUR",
            "DAY",
            "MONTH",
            "DAY_OF_WEEK"
        ]
    ].head(10)
)


# ============================================================
# 18. CHECK DATE-TIME RANGES
# ============================================================

print("\nHOUR range:")

print(
    df["HOUR"].min(),
    "to",
    df["HOUR"].max()
)


print("\nDAY range:")

print(
    df["DAY"].min(),
    "to",
    df["DAY"].max()
)


print("\nMONTH range:")

print(
    df["MONTH"].min(),
    "to",
    df["MONTH"].max()
)


print("\nDAY_OF_WEEK range:")

print(
    df["DAY_OF_WEEK"].min(),
    "to",
    df["DAY_OF_WEEK"].max()
)


# ============================================================
# 19. FINAL DATASET CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREPROCESSED DATASET")
print("=" * 70)


print("\nFinal Shape:")

print(
    df.shape
)


print("\nFinal Columns:")

for column in df.columns:

    print("-", column)


# ============================================================
# 20. FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL MISSING VALUE CHECK")
print("=" * 70)


final_missing = df.isnull().sum()


print(
    final_missing
)


# ============================================================
# 21. FINAL DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL DUPLICATE CHECK")
print("=" * 70)


final_duplicates = df.duplicated().sum()


print(
    "Duplicate rows:",
    final_duplicates
)


# ============================================================
# 22. SAVE FINAL PREPROCESSED DATASET
# ============================================================

df.to_csv(
    final_output_path,
    index=False
)


# ============================================================
# 23. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nFinal preprocessed dataset saved at:")

print(
    final_output_path
)


print("\nFinal Dataset Shape:")

print(
    df.shape
)