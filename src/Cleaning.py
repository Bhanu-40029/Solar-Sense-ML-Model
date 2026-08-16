# ============================================================
# SOLARSENSE - DATA CLEANING
# ============================================================

import pandas as pd
import os


# ============================================================
# 1. PATHS
# ============================================================

input_path = (
    "D:/2nd Year/ML Solor Sense/Datasets/"
    "SolarSense_Day_Night_Balanced_Raw.csv"
)

output_folder = (
    "D:/2nd Year/ML Solor Sense/Datasets"
)

os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(
    output_folder,
    "SolarSense_Cleaned_Data.csv"
)



# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(input_path)

print("=" * 70)
print("SOLARSENSE - DATA CLEANING")
print("=" * 70)

print("\nOriginal Dataset Shape:")
print(df.shape)


# ============================================================
# 3. STORE ORIGINAL INFORMATION
# ============================================================

original_rows = len(df)

cleaning_summary = []


# ============================================================
# 4. DATE_TIME CONVERSION
# ============================================================

print("\n" + "=" * 70)
print("DATE_TIME CHECK")
print("=" * 70)

# Count invalid date values before conversion
invalid_dates_before = pd.to_datetime(
    df["DATE_TIME"],
    dayfirst=True,
    errors="coerce"
).isna().sum()

# Convert DATE_TIME
df["DATE_TIME"] = pd.to_datetime(
    df["DATE_TIME"],
    dayfirst=True,
    errors="coerce"
)

print("Invalid DATE_TIME values:", invalid_dates_before)

# Remove rows where DATE_TIME could not be interpreted
if invalid_dates_before > 0:

    df = df.dropna(
        subset=["DATE_TIME"]
    ).copy()

    print(
        "Removed invalid DATE_TIME rows:",
        invalid_dates_before
    )

else:

    print("No invalid DATE_TIME values found.")


cleaning_summary.append(
    [
        "Invalid DATE_TIME",
        invalid_dates_before
    ]
)


# ============================================================
# 5. MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

missing_before = df.isnull().sum()

print("\nMissing values before cleaning:")
print(missing_before)


# ------------------------------------------------------------
# Handle missing values
# ------------------------------------------------------------
# We don't blindly fill missing values.
# For this project, rows with missing critical values
# are removed because those values are required for ML.

critical_columns = [
    "DATE_TIME",
    "INVERTER_ID",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "AC_POWER"
]

missing_critical_rows = df[
    critical_columns
].isnull().any(axis=1).sum()

if missing_critical_rows > 0:

    df = df.dropna(
        subset=critical_columns
    ).copy()

    print(
        "\nRemoved rows with missing critical values:",
        missing_critical_rows
    )

else:

    print(
        "\nNo missing values found in critical columns."
    )


cleaning_summary.append(
    [
        "Rows with missing critical values",
        missing_critical_rows
    ]
)


# ============================================================
# 6. DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

duplicates_before = df.duplicated().sum()

print(
    "Duplicate rows before cleaning:",
    duplicates_before
)

if duplicates_before > 0:

    df = df.drop_duplicates().copy()

    print(
        "Duplicate rows removed:",
        duplicates_before
    )

else:

    print("No duplicate rows found.")


cleaning_summary.append(
    [
        "Duplicate rows",
        duplicates_before
    ]
)


# ============================================================
# 7. INVALID NEGATIVE VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("INVALID NEGATIVE VALUE CHECK")
print("=" * 70)

# These variables cannot physically have negative values
non_negative_columns = [
    "DC_POWER",
    "AC_POWER",
    "DAILY_YIELD",
    "TOTAL_YIELD",
    "IRRADIATION"
]

negative_summary = {}

for column in non_negative_columns:

    if column in df.columns:

        negative_count = (
            df[column] < 0
        ).sum()

        negative_summary[column] = negative_count

        print(
            f"{column}: {negative_count} negative values"
        )


# ============================================================
# 8. REMOVE ROWS WITH INVALID NEGATIVE VALUES
# ============================================================

invalid_negative_mask = pd.Series(
    False,
    index=df.index
)

for column in non_negative_columns:

    if column in df.columns:

        invalid_negative_mask |= (
            df[column] < 0
        )

invalid_negative_rows = invalid_negative_mask.sum()

if invalid_negative_rows > 0:

    df = df[
        ~invalid_negative_mask
    ].copy()

    print(
        "\nRemoved rows containing invalid negative values:",
        invalid_negative_rows
    )

else:

    print(
        "\nNo invalid negative values found."
    )


cleaning_summary.append(
    [
        "Rows with invalid negative values",
        invalid_negative_rows
    ]
)


# ============================================================
# 9. CHECK REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("COLUMN CHECK")
print("=" * 70)

required_columns = [
    "DATE_TIME",
    "PLANT_ID",
    "INVERTER_ID",
    "DC_POWER",
    "AC_POWER",
    "DAILY_YIELD",
    "TOTAL_YIELD",
    "WEATHER_SENSOR_ID",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "PERIOD"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if len(missing_columns) == 0:

    print("All expected columns are present.")

else:

    print(
        "Missing columns:",
        missing_columns
    )


# ============================================================
# 10. FINAL DUPLICATE CHECK
# ============================================================

final_duplicates = df.duplicated().sum()


# ============================================================
# 11. FINAL MISSING VALUE CHECK
# ============================================================

final_missing = df.isnull().sum().sum()


# ============================================================
# 12. FINAL DATASET INFORMATION
# ============================================================

final_rows = len(df)

rows_removed = (
    original_rows - final_rows
)

print("\n" + "=" * 70)
print("FINAL CLEANING RESULTS")
print("=" * 70)

print(
    "\nOriginal rows:",
    original_rows
)

print(
    "Final rows:",
    final_rows
)

print(
    "Rows removed:",
    rows_removed
)

print(
    "Final columns:",
    len(df.columns)
)

print(
    "Final missing values:",
    final_missing
)

print(
    "Final duplicate rows:",
    final_duplicates
)


# ============================================================
# 13. DAY / NIGHT CHECK
# ============================================================

if "PERIOD" in df.columns:

    print("\n" + "=" * 70)
    print("DAY / NIGHT CHECK")
    print("=" * 70)

    print(
        df["PERIOD"].value_counts()
    )


# ============================================================
# 14. CREATE CLEANING SUMMARY
# ============================================================

cleaning_summary.extend(
    [
        [
            "Original rows",
            original_rows
        ],
        [
            "Final rows",
            final_rows
        ],
        [
            "Total rows removed",
            rows_removed
        ],
        [
            "Final missing values",
            final_missing
        ],
        [
            "Final duplicate rows",
            final_duplicates
        ]
    ]
)

cleaning_summary_df = pd.DataFrame(
    cleaning_summary,
    columns=[
        "Cleaning Operation",
        "Records Affected"
    ]
)




# ============================================================
# 15. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCleaned dataset saved at:")
print(output_path)

print("\nNext stage:")
print("Feature Engineering / Preprocessing")