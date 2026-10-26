
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned battery data
df = pd.read_csv("battery_cleaned.csv")

# Plot 1: Battery State of Health over cycles
plt.figure(figsize=(10, 5))

for battery in df["Battery"].unique():
    part = df[df["Battery"] == battery]
    plt.plot(part["Cycle"], part["SoH"], label=battery)

plt.axhline(80, linestyle="--", label="80% threshold")
plt.axhline(60, linestyle="--", label="60% threshold")

plt.xlabel("Cycle Number")
plt.ylabel("State of Health (%)")
plt.title("Battery Degradation over Cycles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("battery_soh.png")
plt.show()

# Plot 2: Health class distribution
df["Health"].value_counts().reindex(
    ["Healthy", "Degrading", "Critical"], fill_value=0
).plot(kind="bar", rot=0)

plt.title("Battery Health Class Distribution")
plt.xlabel("Health Class")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("health_distribution.png")
plt.show()

print("Graphs saved successfully!")
print("\nCritical records:")
print(df[df["Health"] == "Critical"].to_string(index=False))
