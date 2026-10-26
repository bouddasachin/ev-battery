
from scipy.io import loadmat
from pathlib import Path

# Load NASA battery data
file_path = Path("data") / "B0005.mat"
data = loadmat(file_path)

battery = data["B0005"][0, 0]
cycles = battery["cycle"][0]

print("Total cycles:", len(cycles))
print("\nFirst 5 cycles:\n")

for cycle in cycles[:5]:
    cycle_type = cycle["type"][0]
    cycle_type = cycle_type.item()

    print("Cycle type:", cycle_type)

    if cycle_type == "discharge":
        capacity = cycle["data"][0, 0]["Capacity"][0, 0]
        print("Capacity:", float(capacity))

    print("-" * 30)
