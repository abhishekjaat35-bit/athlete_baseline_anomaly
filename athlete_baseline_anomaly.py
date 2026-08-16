import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("=" * 80)
print("          ATHLETE BASELINE & ANOMALY DETECTION SYSTEM")
print("=" * 80)


# ------------------------------------------
# Load Data
# ------------------------------------------

data = pd.read_csv(
    "athlete_baseline_data.csv"
)

data["Date"] = pd.to_datetime(
    data["Date"]
)

data = data.sort_values(
    ["Athlete", "Date"]
).reset_index(drop=True)


# ------------------------------------------
# Data Validation
# ------------------------------------------

print("\n" + "=" * 80)
print("DATA VALIDATION")
print("=" * 80)

print(f"Rows           : {len(data)}")
print(f"Columns        : {len(data.columns)}")
print(
    f"Missing values : "
    f"{data.isnull().sum().sum()}"
)
print(
    f"Athletes       : "
    f"{data['Athlete'].nunique()}"
)


# ------------------------------------------
# Athlete Baselines
# ------------------------------------------

data["Load_Baseline"] = (
    data.groupby("Athlete")["Training_Load"]
    .transform("mean")
)

data["Load_SD"] = (
    data.groupby("Athlete")["Training_Load"]
    .transform("std")
)

data["Readiness_Baseline"] = (
    data.groupby("Athlete")["Readiness_Score"]
    .transform("mean")
)

data["Readiness_SD"] = (
    data.groupby("Athlete")["Readiness_Score"]
    .transform("std")
)


# ------------------------------------------
# Calculate Z-Scores
# ------------------------------------------

data["Load_Z_Score"] = (
    (
        data["Training_Load"]
        -
        data["Load_Baseline"]
    )
    /
    data["Load_SD"]
)

data["Readiness_Z_Score"] = (
    (
        data["Readiness_Score"]
        -
        data["Readiness_Baseline"]
    )
    /
    data["Readiness_SD"]
)


# ------------------------------------------
# Handle Zero Standard Deviation
# ------------------------------------------

data["Load_Z_Score"] = (
    data["Load_Z_Score"]
    .replace(
        [np.inf, -np.inf],
        0
    )
    .fillna(0)
)

data["Readiness_Z_Score"] = (
    data["Readiness_Z_Score"]
    .replace(
        [np.inf, -np.inf],
        0
    )
    .fillna(0)
)


# ------------------------------------------
# Load Anomaly Classification
# ------------------------------------------

def classify_anomaly(z_score):

    absolute_z = abs(z_score)

    if absolute_z >= 2.0:
        return "ANOMALY"

    elif absolute_z >= 1.5:
        return "WATCH"

    else:
        return "NORMAL"


data["Load_Status"] = (
    data["Load_Z_Score"]
    .apply(classify_anomaly)
)

data["Readiness_Status"] = (
    data["Readiness_Z_Score"]
    .apply(classify_anomaly)
)


# ------------------------------------------
# Overall Monitoring Status
# ------------------------------------------

def overall_status(row):

    if (
        row["Load_Status"] == "ANOMALY"
        or
        row["Readiness_Status"] == "ANOMALY"
    ):
        return "ANOMALY"

    elif (
        row["Load_Status"] == "WATCH"
        or
        row["Readiness_Status"] == "WATCH"
    ):
        return "WATCH"

    else:
        return "NORMAL"


data["Overall_Status"] = data.apply(
    overall_status,
    axis=1
)


# ------------------------------------------
# Monitoring Action
# ------------------------------------------

def monitoring_action(status):

    if status == "ANOMALY":

        return (
            "Review athlete response, "
            "training load and recovery."
        )

    elif status == "WATCH":

        return (
            "Continue monitoring and "
            "review recent trends."
        )

    else:

        return (
            "Continue normal monitoring."
        )


data["Monitoring_Action"] = (
    data["Overall_Status"]
    .apply(monitoring_action)
)


# ------------------------------------------
# Display Results
# ------------------------------------------

print("\n" + "=" * 80)
print("ANOMALY DETECTION RESULTS")
print("=" * 80)

display_columns = [
    "Athlete",
    "Date",
    "Training_Load",
    "Load_Baseline",
    "Load_Z_Score",
    "Load_Status",
    "Readiness_Score",
    "Readiness_Baseline",
    "Readiness_Z_Score",
    "Readiness_Status",
    "Overall_Status"
]

display_data = data[display_columns].copy()

display_data["Load_Baseline"] = (
    display_data["Load_Baseline"].round(1)
)

display_data["Load_Z_Score"] = (
    display_data["Load_Z_Score"].round(2)
)

display_data["Readiness_Baseline"] = (
    display_data["Readiness_Baseline"].round(1)
)

display_data["Readiness_Z_Score"] = (
    display_data["Readiness_Z_Score"].round(2)
)

print(
    display_data.to_string(
        index=False
    )
)


# ------------------------------------------
# Athlete Baseline Summary
# ------------------------------------------

baseline_summary = (
    data.groupby("Athlete")
    .agg(
        Observations=(
            "Athlete",
            "count"
        ),

        Average_Load=(
            "Training_Load",
            "mean"
        ),

        Load_SD=(
            "Training_Load",
            "std"
        ),

        Average_Readiness=(
            "Readiness_Score",
            "mean"
        ),

        Readiness_SD=(
            "Readiness_Score",
            "std"
        ),

        Load_Anomalies=(
            "Load_Status",
            lambda x:
            (x == "ANOMALY").sum()
        ),

        Readiness_Anomalies=(
            "Readiness_Status",
            lambda x:
            (x == "ANOMALY").sum()
        ),

        Total_Anomalies=(
            "Overall_Status",
            lambda x:
            (x == "ANOMALY").sum()
        )
    )
    .reset_index()
)


print("\n" + "=" * 80)
print("ATHLETE BASELINE SUMMARY")
print("=" * 80)

print(
    baseline_summary.to_string(
        index=False,
        formatters={
            "Average_Load":
                "{:.1f}".format,

            "Load_SD":
                "{:.1f}".format,

            "Average_Readiness":
                "{:.1f}".format,

            "Readiness_SD":
                "{:.1f}".format
        }
    )
)


# ------------------------------------------
# Detected Anomalies
# ------------------------------------------

anomalies = data[
    data["Overall_Status"] == "ANOMALY"
].copy()


print("\n" + "=" * 80)
print("DETECTED ANOMALIES")
print("=" * 80)

if len(anomalies) == 0:

    print("No major anomalies detected.")

else:

    for _, row in anomalies.iterrows():

        print(
            f"{row['Athlete']:<10} "
            f"{row['Date'].date()} | "
            f"Load: {row['Training_Load']:>4.0f} AU | "
            f"Load Z: {row['Load_Z_Score']:>5.2f} | "
            f"Readiness: {row['Readiness_Score']:>3.0f}% | "
            f"Readiness Z: "
            f"{row['Readiness_Z_Score']:>5.2f}"
        )


# ------------------------------------------
# Training Load Anomaly Plot
# ------------------------------------------

plt.figure(
    figsize=(11, 6)
)

for athlete in data["Athlete"].unique():

    athlete_data = data[
        data["Athlete"] == athlete
    ]

    plt.plot(
        athlete_data["Date"],
        athlete_data["Training_Load"],
        marker="o",
        label=athlete
    )

plt.title(
    "Training Load and Individual Athlete Baselines"
)

plt.xlabel("Date")

plt.ylabel(
    "Training Load (AU)"
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "training_load_anomalies.png",
    dpi=300
)

plt.show()


# ------------------------------------------
# Readiness Anomaly Plot
# ------------------------------------------

plt.figure(
    figsize=(11, 6)
)

for athlete in data["Athlete"].unique():

    athlete_data = data[
        data["Athlete"] == athlete
    ]

    plt.plot(
        athlete_data["Date"],
        athlete_data["Readiness_Score"],
        marker="o",
        label=athlete
    )

plt.title(
    "Readiness and Individual Athlete Baselines"
)

plt.xlabel("Date")

plt.ylabel(
    "Readiness Score (%)"
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "readiness_anomalies.png",
    dpi=300
)

plt.show()


# ------------------------------------------
# Export Results
# ------------------------------------------

data.to_csv(
    "athlete_anomaly_results.csv",
    index=False
)

baseline_summary.to_csv(
    "athlete_baseline_summary.csv",
    index=False
)


# ------------------------------------------
# Final Output
# ------------------------------------------

print("\n" + "=" * 80)
print("ANOMALY DETECTION COMPLETE")
print("=" * 80)

print("Generated files:")

print("1. athlete_anomaly_results.csv")
print("2. athlete_baseline_summary.csv")
print("3. training_load_anomalies.png")
print("4. readiness_anomalies.png")

print("\n" + "=" * 80)
print("BASELINE • DETECT • MONITOR • RESPOND")
print("=" * 80)