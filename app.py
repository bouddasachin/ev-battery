
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============== PAGE CONFIG ===============
st.set_page_config(
    page_title="EV BatteryGuard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).parent
DATA_FILE = BASE / "battery_cleaned.csv"

# =============== DARK BLUE THEME ===============
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#020b18,#031629);
    color: #e5efff;
}
[data-testid="stSidebar"] {
    background: #03101f;
    border-right: 1px solid #12314b;
}
.block-container {
    padding: 1.1rem 1.2rem 2rem;
    max-width: 1800px;
}
h1,h2,h3 { color:#eef6ff !important; }
.metric-card,.panel {
    background: linear-gradient(145deg,#071b30,#071322);
    border: 1px solid #123a55;
    border-radius: 13px;
    padding: 15px;
}
.metric-card { min-height: 120px; }
.metric-label { color:#a9c2dc; font-size:12px; margin-bottom:11px; }
.metric-value { color:#f3f8ff; font-size:25px; font-weight:700; }
.metric-sub { color:#00e99a; font-size:12px; margin-top:8px; }
.small-muted { color:#8ca9c7; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# =============== LOAD DATA ===============
if not DATA_FILE.exists():
    st.error("battery_cleaned.csv was not found next to app.py.")
    st.stop()

df = pd.read_csv(DATA_FILE)

required = ["Battery", "Cycle", "Capacity"]
if not all(c in df.columns for c in required):
    st.error("CSV must contain Battery, Cycle, and Capacity.")
    st.stop()

for col in ["Cycle", "Capacity", "SoH", "Voltage", "Current", "Temperature"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Battery", "Cycle", "Capacity"]).copy()
df = df[(df["Cycle"] > 0) & (df["Capacity"] > 0)].copy()

if df.empty:
    st.error("No valid battery records found.")
    st.stop()

# Calculate SoH only if the CSV doesn't contain it.
if "SoH" not in df.columns:
    initial_capacity = df.groupby("Battery")["Capacity"].transform("first")
    df["SoH"] = df["Capacity"] / initial_capacity * 100

def classify(soh):
    if pd.isna(soh):
        return "Unavailable"
    if soh >= 75:
        return "Healthy"
    if soh >= 70:
        return "Degrading"
    return "Critical"

df["Health"] = df["SoH"].apply(classify)

# Keep source IDs internally, but never display them.
battery_keys = sorted(df["Battery"].unique(), key=str)
name_map = {
    key: f"Battery {i + 1}"
    for i, key in enumerate(battery_keys)
}
df["Display Battery"] = df["Battery"].map(name_map)

latest = (
    df.sort_values("Cycle")
      .groupby("Battery", sort=False)
      .tail(1)
      .copy()
)
latest["Display Battery"] = latest["Battery"].map(name_map)
latest["Health"] = latest["SoH"].apply(classify)

avg_soh = latest["SoH"].mean()
avg_capacity = latest["Capacity"].mean()
total_records = len(df)
total_cycles = int(df.groupby("Battery")["Cycle"].nunique().sum())
overall_status = classify(avg_soh)

def latest_average(column):
    if column not in latest.columns:
        return None
    values = latest[column].dropna()
    return float(values.mean()) if not values.empty else None

avg_voltage = latest_average("Voltage")
avg_current = latest_average("Current")
avg_temperature = latest_average("Temperature")

def fmt(value, unit="", decimals=2):
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{value:.{decimals}f}{unit}"

def metric_card(label, value, subtitle):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

# =============== CHART FUNCTION ===============
def plot_all(column, title, ylabel, thresholds=None):
    st.markdown(f"#### {title}")

    if column not in df.columns or df[column].notna().sum() == 0:
        st.info(f"{title}: measurement is unavailable in the dataset.")
        return

    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor("#06172a")
    ax.set_facecolor("#06172a")

    for i, key in enumerate(battery_keys, start=1):
        group = df[df["Battery"] == key].sort_values("Cycle")
        group = group.dropna(subset=[column])
        if group.empty:
            continue
        ax.plot(
            group["Cycle"], group[column],
            linewidth=1.8, label=f"Battery {i}"
        )

    if thresholds:
        for value, label, style in thresholds:
            ax.axhline(value, linestyle=style, linewidth=1, label=label)

    ax.set_xlabel("Recorded Cycle", color="#a9c2dc")
    ax.set_ylabel(ylabel, color="#a9c2dc")
    ax.tick_params(colors="#a9c2dc")
    ax.grid(alpha=0.15)

    for spine in ax.spines.values():
        spine.set_color("#18354e")

    ax.legend(
        fontsize=8, ncol=4, frameon=False,
        facecolor="#06172a", labelcolor="#d9eaff"
    )
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# =============== SIDEBAR ===============
with st.sidebar:
    st.markdown("## 🔋 EV BatteryGuard")
    st.caption("EARLY DETECTION OF BATTERY FAILURE")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Battery Details", "Prediction",
         "Analytics", "Dataset", "Model Performance"]
    )

    st.divider()
    st.caption("Source: Historical battery dataset")
    st.caption("Live sensors: Not connected")
    st.caption("Healthy: SoH ≥ 75%")
    st.caption("Degrading: 70% to below 75%")
    st.caption("Critical: SoH below 70%")

st.title("EV BatteryGuard")
st.markdown(
    '<p class="small-muted">Battery health monitoring and degradation analysis</p>',
    unsafe_allow_html=True
)

# =============== OVERVIEW ===============
if page == "Overview":
    cards = st.columns(5)
    card_data = [
        ("VOLTAGE", fmt(avg_voltage, " V"), "Latest records average"),
        ("TEMPERATURE", fmt(avg_temperature, " °C"), "Latest records average"),
        ("CHARGING CYCLES", f"{total_cycles:,}", "Across all profiles"),
        ("CAPACITY", fmt(avg_capacity, " Ah"), "Latest records average"),
        ("STATE OF HEALTH", fmt(avg_soh, "%", 1), overall_status)
    ]

    for col, (label, value, subtitle) in zip(cards, card_data):
        with col:
            metric_card(label, value, subtitle)

    st.write("")
    left, middle, right = st.columns([1.5, 1, 1])

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Present Battery Details")
        st.caption("Summary across all available battery profiles")

        for label, value in [
            ("Voltage", fmt(avg_voltage, " V")),
            ("Current", fmt(avg_current, " A")),
            ("Temperature", fmt(avg_temperature, " °C")),
            ("Charging cycles", f"{total_cycles:,}"),
            ("Capacity", fmt(avg_capacity, " Ah")),
            ("State of Health", fmt(avg_soh, "%", 1))
        ]:
            st.markdown(
                f"<div style='background:#091d32;padding:8px 11px;"
                f"border-radius:7px;margin-bottom:6px;display:flex;"
                f"justify-content:space-between;gap:10px;'>"
                f"<span style='color:#b8cde2'>{label}</span>"
                f"<b style='color:#f1f7ff'>{value}</b></div>",
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    with middle:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Battery Health Prediction")
        st.markdown(
            f"<div style='font-size:27px;font-weight:bold;color:"
            f"{'#00ed9a' if overall_status == 'Healthy' else '#ffb020' if overall_status == 'Degrading' else '#ff5268'}'>"
            f"{overall_status}</div>",
            unsafe_allow_html=True
        )
        st.metric("Average SoH", fmt(avg_soh, "%", 1))
        st.progress(max(0.0, min(1.0, float(avg_soh) / 100)))
        st.caption("Classification based on project-defined SoH thresholds.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Battery Status Indicator")

        fig, ax = plt.subplots(figsize=(3, 2.4))
        fig.patch.set_facecolor("#06172a")
        ax.pie(
            [max(0, min(100, avg_soh)), max(0, 100 - min(100, avg_soh))],
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.24, "edgecolor": "#06172a"},
            colors=["#00d889", "#15314a"]
        )
        ax.text(
            0, 0, fmt(avg_soh, "%", 1),
            ha="center", va="center", fontsize=18,
            fontweight="bold", color="#f3f8ff"
        )
        ax.axis("equal")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        counts = latest["Health"].value_counts()
        st.write(f"🟢 Healthy: {counts.get('Healthy', 0)}")
        st.write(f"🟠 Degrading: {counts.get('Degrading', 0)}")
        st.write(f"🔴 Critical: {counts.get('Critical', 0)}")
        st.markdown('</div>', unsafe_allow_html=True)

    # THE ONLY SoH GRAPH ON OVERVIEW
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    plot_all(
        "SoH",
        "All-Battery SoH Comparison",
        "SoH (%)",
        [(75, "Healthy limit", "--"), (70, "Critical limit", ":")]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        plot_all("Voltage", "Voltage Trend — All Batteries", "Voltage (V)")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        plot_all("Temperature", "Temperature Trend — All Batteries", "Temperature (°C)")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    plot_all("Capacity", "Capacity Degradation — All Batteries", "Capacity (Ah)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Recent Battery Health Results")

    recent = latest.sort_values("Cycle", ascending=False).head(10).reset_index(drop=True)
    table = pd.DataFrame({
        "S.No": range(1, len(recent) + 1),
        "Voltage (V)": recent["Voltage"] if "Voltage" in recent else None,
        "Current (A)": recent["Current"] if "Current" in recent else None,
        "Temperature (°C)": recent["Temperature"] if "Temperature" in recent else None,
        "Cycles": recent["Cycle"].astype(int),
        "Capacity (Ah)": recent["Capacity"],
        "SoH (%)": recent["SoH"].round(1),
        "Health Status": recent["Health"]
    })
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("Health status uses the configured SoH thresholds, not a verified ML prediction.")
    st.markdown('</div>', unsafe_allow_html=True)

# =============== BATTERY DETAILS ===============
elif page == "Battery Details":
    st.header("Battery Details — All Profiles")
    plot_all("Capacity", "Capacity Comparison", "Capacity (Ah)")
    plot_all("SoH", "SoH Comparison", "SoH (%)")
    plot_all("Voltage", "Voltage Comparison", "Voltage (V)")
    plot_all("Current", "Current Comparison", "Current (A)")
    plot_all("Temperature", "Temperature Comparison", "Temperature (°C)")

# =============== PREDICTION ===============
elif page == "Prediction":
    st.header("Battery Health Classification")
    st.info(
        "This page currently shows threshold-based health labels. "
        "Connect the saved model and its exact training features to display "
        "verified machine-learning predictions."
    )

    result = latest[["Cycle", "Capacity", "SoH", "Health"]].reset_index(drop=True)
    result.insert(0, "S.No", range(1, len(result) + 1))
    st.dataframe(result, use_container_width=True, hide_index=True)

# =============== ANALYTICS ===============
elif page == "Analytics":
    st.header("Analytics and Reports")
    st.metric("Average SoH", fmt(avg_soh, "%", 1))
    st.metric("Average Capacity", fmt(avg_capacity, " Ah"))
    st.metric("Total Records", f"{total_records:,}")

    # SoH graph is allowed here; it is not duplicated on Overview.
    plot_all("SoH", "SoH Trend — All Batteries", "SoH (%)")
    plot_all("Capacity", "Capacity Trend — All Batteries", "Capacity (Ah)")
    plot_all("Voltage", "Voltage Trend — All Batteries", "Voltage (V)")
    plot_all("Current", "Current Trend — All Batteries", "Current (A)")
    plot_all("Temperature", "Temperature Trend — All Batteries", "Temperature (°C)")

# =============== DATASET ===============
elif page == "Dataset":
    st.header("Dataset Explorer")
    columns = ["Cycle", "Capacity", "SoH", "Health"]
    for col in ["Voltage", "Current", "Temperature"]:
        if col in df.columns:
            columns.append(col)

    view = df[columns].copy()
    st.write(f"Total records: {len(view):,}")
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "Download displayed data",
        view.to_csv(index=False).encode("utf-8"),
        file_name="battery_dataset_view.csv",
        mime="text/csv"
    )

# =============== MODEL PERFORMANCE ===============
elif page == "Model Performance":
    st.header("Model Performance")
    model_path = BASE / "battery_model.pkl"

    if model_path.exists():
        st.success("Saved model file found: battery_model.pkl")
        st.write(
            "Show accuracy, precision, recall, F1-score and a confusion matrix "
            "from the actual training/test evaluation in train_model.py."
        )
    else:
        st.warning("battery_model.pkl was not found.")

    st.info(
        "Metrics are not invented here. They should be loaded from the "
        "model's actual evaluation results."
    )

st.divider()
st.caption(
    "EV BatteryGuard | Historical dataset analysis | "
    "No live sensor connection assumed"
)
