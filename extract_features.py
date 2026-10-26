
from scipy.io import loadmat
from pathlib import Path
import pandas as pd

# Automatically find the project and data folder
base_folder = Path(__file__).parent
data_folder = base_folder / "data"

# Find every MATLAB battery file
battery_files = sorted(data_folder.glob("*.mat"))

if not battery_files:
    print("No .mat files found in the data folder.")
    raise SystemExit(1)

rows = []

# Process every battery file
for file_path in battery_files:
    print(f"Processing {file_path.name}...")

    try:
        data = loadmat(file_path)
        battery_name = file_path.stem

        if battery_name not in data:
            print(f"Skipping {file_path.name}: expected battery data not found.")
            continue

        battery = data[battery_name][0, 0]
        cycles = battery["cycle"][0]

        for cycle_number, cycle in enumerate(cycles, start=1):
            cycle_type = cycle["type"][0].item()

            if cycle_type != "discharge":
                continue

            measurement = cycle["data"][0, 0]
            capacity = float(measurement["Capacity"][0, 0])

            rows.append({
                "Battery": battery_name,
                "Cycle": cycle_number,
                "Capacity": capacity
            })

    except Exception as error:
        print(f"Could not process {file_path.name}: {error}")

# Save extracted records
if not rows:
    print("No discharge records extracted. Check the .mat files.")
    raise SystemExit(1)

df = pd.DataFrame(rows)
output_path = base_folder / "battery_features.csv"
df.to_csv(output_path, index=False)

print("\nExtraction completed!")
print("Battery files found:", len(battery_files))
print("Total discharge records:", len(df))
print("\nRecords per battery:")
print(df.groupby("Battery").size())
print("\nSaved to:", output_path)
