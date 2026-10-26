
from pathlib import Path

data_folder = Path("data")

files = ["B0005.mat", "B0006.mat", "B0007.mat"]

for name in files:
    path = data_folder / name

    if path.exists():
        print(f"FOUND: {name}")
    else:
        print(f"MISSING: {name}")
