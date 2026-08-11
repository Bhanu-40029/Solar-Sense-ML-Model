import pandas as pd

# --------------------------------
# 1. LOAD DATASETS
# --------------------------------

generation = pd.read_csv( "D:/2nd Year/ML Solor Sense/Datasets/Plant_1_Generation_Data.csv")

weather = pd.read_csv("D:/2nd Year/ML Solor Sense/Datasets/Plant_1_Weather_Sensor_Data.csv")


# --------------------------------
# 2. CONVERT DATE_TIME
# --------------------------------

generation["DATE_TIME"] = pd.to_datetime(
    generation["DATE_TIME"],
    dayfirst=True
)

weather["DATE_TIME"] = pd.to_datetime(
    weather["DATE_TIME"],
    dayfirst=True
)


# --------------------------------
# 3. RENAME SOURCE_KEY
# --------------------------------

generation.rename(
    columns={"SOURCE_KEY": "INVERTER_ID"},
    inplace=True
)

weather.rename(
    columns={"SOURCE_KEY": "WEATHER_SENSOR_ID"},
    inplace=True
)


# --------------------------------
# 4. RENAME WEATHER SENSOR
# --------------------------------

weather["WEATHER_SENSOR_ID"] = "WEATHER_SENSOR_01"


# --------------------------------
# 5. RENAME INVERTER IDs
# --------------------------------

# Get unique inverter IDs
unique_inverters = generation["INVERTER_ID"].unique()

# Create mapping:
# Original ID -> INV_01, INV_02, ...
inverter_mapping = {
    old_id: f"INV_{i+1:02d}"
    for i, old_id in enumerate(unique_inverters)
}

# Replace original IDs
generation["INVERTER_ID"] = generation["INVERTER_ID"].map(
    inverter_mapping
)

print("\nInverter Mapping:")
print(inverter_mapping)


# --------------------------------
# 6. MERGE DATASETS
# --------------------------------

merged_data = pd.merge(
    generation,
    weather,
    on=["DATE_TIME", "PLANT_ID"],
    how="inner"
)

print("\nBefore limiting records:")
print(merged_data.shape)


# --------------------------------
# 7. KEEP 50,000 RECORDS
# --------------------------------

merged_data = merged_data.head(50000)

print("\nAfter limiting records:")
print(merged_data.shape)


# --------------------------------
# 8. CHECK MERGED DATA
# --------------------------------

print("\nGeneration Shape:")
print(generation.shape)

print("\nWeather Shape:")
print(weather.shape)

print("\nMerged Shape:")
print(merged_data.shape)

print("\nMerged Columns:")
print(merged_data.columns)

print("\nFirst 5 rows:")
print(merged_data.head())


# --------------------------------
# 9. SAVE MERGED DATASET
# --------------------------------

output_path = (
    "D:/2nd Year/ML Solor Sense/Datasets/Plant_1_Merged_Data.csv"
)

merged_data.to_csv(
    output_path,
    index=False
)

print("\nMerged successfully!")
print("Saved at:", output_path)