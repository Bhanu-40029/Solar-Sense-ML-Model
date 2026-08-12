# ============================================================
# SOLARSENSE - DATA UNDERSTANDING
# ============================================================

import pandas as pd


# ============================================================
# 1. LOAD MERGED DATASET
# ============================================================

file_path = (
    "D:/2nd Year/ML Solor Sense/Datasets/"
    "SolarSense_Day_Night_Balanced_Raw.csv"
)

df = pd.read_csv(file_path)

print("=" * 70)
print("SOLARSENSE - DATASET UNDERSTANDING")
print("=" * 70)

print("\nDataset loaded successfully!")


# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET SHAPE")
print("=" * 70)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])
print("Shape   :", df.shape)


# ============================================================
# 3. COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("2. COLUMN NAMES")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# ============================================================
# 4. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# 5. FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 70)
print("4. FIRST 5 ROWS")
print("=" * 70)

print(df.head())


# ============================================================
# 6. LAST 5 ROWS
# ============================================================

print("\n" + "=" * 70)
print("5. LAST 5 ROWS")
print("=" * 70)

print(df.tail())


# ============================================================
# 7. COMPLETE DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("6. DATASET INFORMATION")
print("=" * 70)

df.info()


# ============================================================
# 8. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("7. STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe())


# ============================================================
# 9. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("8. MISSING VALUES")
print("=" * 70)

missing_values = df.isnull().sum()

print(missing_values)


# ============================================================
# 10. MISSING VALUE PERCENTAGE
# ============================================================

print("\n" + "=" * 70)
print("9. MISSING VALUE PERCENTAGE")
print("=" * 70)

missing_percentage = (
    df.isnull().sum() / len(df) * 100
)

print(missing_percentage.round(2))


# ============================================================
# 11. DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 70)
print("10. DUPLICATE RECORDS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


# ============================================================
# 12. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("11. UNIQUE VALUES")
print("=" * 70)

print("Number of Plants:",
      df["PLANT_ID"].nunique())

print("Number of Inverters:",
      df["INVERTER_ID"].nunique())

print("Number of Weather Sensors:",
      df["WEATHER_SENSOR_ID"].nunique())


# ============================================================
# 13. INVERTER IDs
# ============================================================

print("\n" + "=" * 70)
print("12. INVERTER IDs")
print("=" * 70)

print(df["INVERTER_ID"].unique())


# ============================================================
# 14. WEATHER SENSOR IDs
# ============================================================

print("\n" + "=" * 70)
print("13. WEATHER SENSOR IDs")
print("=" * 70)

print(df["WEATHER_SENSOR_ID"].unique())


# ============================================================
# 15. PLANT IDs
# ============================================================

print("\n" + "=" * 70)
print("14. PLANT IDs")
print("=" * 70)

print(df["PLANT_ID"].unique())


# ============================================================
# 16. DATE AND TIME ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("15. DATE AND TIME ANALYSIS")
print("=" * 70)

df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

print("Starting Date :", df["DATE_TIME"].min())
print("Ending Date   :", df["DATE_TIME"].max())

print(
    "Total Duration:",
    df["DATE_TIME"].max() - df["DATE_TIME"].min()
)


# ============================================================
# 17. NUMERICAL COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("16. NUMERICAL COLUMNS")
print("=" * 70)

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print(list(numerical_columns))


# ============================================================
# 18. CATEGORICAL / OBJECT COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("17. CATEGORICAL / OBJECT COLUMNS")
print("=" * 70)

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

print(list(categorical_columns))


# ============================================================
# 19. MINIMUM VALUES
# ============================================================

print("\n" + "=" * 70)
print("18. MINIMUM VALUES")
print("=" * 70)

print(df[numerical_columns].min())


# ============================================================
# 20. MAXIMUM VALUES
# ============================================================

print("\n" + "=" * 70)
print("19. MAXIMUM VALUES")
print("=" * 70)

print(df[numerical_columns].max())


# ============================================================
# 21. MEAN VALUES
# ============================================================

print("\n" + "=" * 70)
print("20. MEAN VALUES")
print("=" * 70)

print(df[numerical_columns].mean())


# ============================================================
# 22. MEDIAN VALUES
# ============================================================

print("\n" + "=" * 70)
print("21. MEDIAN VALUES")
print("=" * 70)

print(df[numerical_columns].median())


# ============================================================
# 23. ZERO VALUES
# ============================================================

print("\n" + "=" * 70)
print("22. ZERO VALUES")
print("=" * 70)

for column in numerical_columns:
    zero_count = (df[column] == 0).sum()
    print(f"{column}: {zero_count}")


# ============================================================
# 24. NEGATIVE VALUES
# ============================================================

print("\n" + "=" * 70)
print("23. NEGATIVE VALUES")
print("=" * 70)

for column in numerical_columns:
    negative_count = (df[column] < 0).sum()
    print(f"{column}: {negative_count}")


# ============================================================
# 25. CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("24. CORRELATION WITH AC_POWER")
print("=" * 70)

correlation = df[numerical_columns].corr()

print(
    correlation["AC_POWER"]
    .sort_values(ascending=False)
)


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SOLARSENSE DATASET SUMMARY")
print("=" * 70)

print("Total Rows          :", df.shape[0])
print("Total Columns       :", df.shape[1])
print("Number of Inverters :", df["INVERTER_ID"].nunique())
print("Number of Plants    :", df["PLANT_ID"].nunique())
print("Weather Sensors     :", df["WEATHER_SENSOR_ID"].nunique())
print("Duplicate Rows      :", df.duplicated().sum())
print("Missing Values      :", df.isnull().sum().sum())
print("Start Date          :", df["DATE_TIME"].min())
print("End Date            :", df["DATE_TIME"].max())

print("\n" + "=" * 70)
print("DATASET UNDERSTANDING COMPLETED")
print("=" * 70)