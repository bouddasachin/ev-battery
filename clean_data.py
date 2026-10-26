
import pandas as pd
from pathlib import Path

# Load extracted battery data
base = Path(__file__).parent
input_file = base / "battery_features.csv"
output_file = base / "battery_cleaned.csv"

df = pd.read_csv(input_file)

print("Original records:", len(df))

# Remove duplicates and clean values
df = df.drop_duplicates()
df["Cycle"] = pd.to_numeric(df["Cycle"], errors="coerce")
df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")

df = df.dropna(subset=["Battery", "Cycle", "Capacity"])
df = df[(df["Cycle"] > 0) & (df["Capacity"] > 0)]

# Calculate SoH relative to each battery's first recorded capacity
first_capacity = df.groupby("Battery")["Capacity"].transform("first")
df["SoH"] = (df["Capacity"] / first_capacity) * 100

# Project classification rules
df["Health"] = df["SoH"].apply(
    lambda x: "Healthy" if x >= 75
    else "Degrading" if x >= 65
    else "Critical"

)

# Remove any invalid classifications
df = df[df["Health"] != "Invalid"]

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("Cleaned records:", len(df))
print("\nHealth classification counts:")
print(df["Health"].value_counts())
print("\nSaved cleaned data to:", output_file)
