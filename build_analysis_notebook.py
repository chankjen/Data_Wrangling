import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SENSOR_CSV = ROOT / "week2_sensor_readings.csv"
HOSPITAL_CSV = ROOT / "hospital_wait_times.csv"
NOTEBOOK_PATH = ROOT / "comprehensive_data_analysis.ipynb"


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


def markdown_table(df, index=True):
    table = df.copy()
    if index:
        table = table.reset_index()
        first_name = table.columns[0]
        if first_name in ("index", None, ""):
            table = table.rename(columns={first_name: ""})
    else:
        table = table.reset_index(drop=True)

    columns = [str(col) for col in table.columns]
    rows = []
    for _, row in table.iterrows():
        rows.append([str(value) for value in row.tolist()])

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def sensor_analysis():
    df = pd.read_csv(SENSOR_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    df["location_clean"] = df["location"].replace(
        {"mombasa": "Mombasa", "MBA": "Mombasa", "NRB": "Nairobi"}
    )
    df["sensor_clean"] = df["sensor_id"].replace(
        {"P1": "Pressure_01", "TEMP_SENS": "temp_sensor"}
    )
    q1 = df["reading"].quantile(0.25)
    q3 = df["reading"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df["reading"] < lower) | (df["reading"] > upper)]

    missing = df.isna().sum().to_frame("missing_values")
    profile = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "start timestamp",
                "end timestamp",
                "duplicate rows",
                "reading mean",
                "reading median",
                "reading std dev",
                "reading min",
                "reading max",
                "IQR lower fence",
                "IQR upper fence",
                "IQR outlier count",
            ],
            "value": [
                f"{df.shape[0]:,}",
                f"{df.shape[1]:,}",
                str(df["timestamp"].min()),
                str(df["timestamp"].max()),
                f"{df.duplicated().sum():,}",
                f"{df['reading'].mean():.2f}",
                f"{df['reading'].median():.2f}",
                f"{df['reading'].std():.2f}",
                f"{df['reading'].min():.2f}",
                f"{df['reading'].max():.2f}",
                f"{lower:.2f}",
                f"{upper:.2f}",
                f"{len(outliers):,}",
            ],
        }
    )

    location_raw = df["location"].value_counts().rename_axis("raw_location").to_frame("rows")
    location_clean = df["location_clean"].value_counts().rename_axis("clean_location").to_frame("rows")
    sensor_raw = df["sensor_id"].value_counts().rename_axis("raw_sensor").to_frame("rows")
    sensor_clean = df["sensor_clean"].value_counts().rename_axis("clean_sensor").to_frame("rows")
    shift_summary = df.groupby("shift")["reading"].agg(["count", "mean", "median", "std"]).round(2)
    clean_combo = (
        df.groupby(["location_clean", "sensor_clean"])["reading"]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )
    hourly = df.groupby("hour")["reading"].agg(["count", "mean"]).round(2)

    return {
        "profile": profile,
        "missing": missing,
        "location_raw": location_raw,
        "location_clean": location_clean,
        "sensor_raw": sensor_raw,
        "sensor_clean": sensor_clean,
        "shift_summary": shift_summary,
        "clean_combo": clean_combo,
        "hourly": hourly,
        "outlier_examples": outliers[["timestamp", "sensor_id", "reading", "location", "shift"]].head(8).round(2),
    }


def hospital_analysis():
    df = pd.read_csv(HOSPITAL_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["ward_clean"] = df["ward_location"].replace(
        {"ER": "Emergency", "emergency": "Emergency", "Peds": "Pediatrics", "GW": "General Ward"}
    )
    df["priority_clean"] = df["priority"].replace({"urgent": "Urgent", "URGENT": "Urgent"})
    q1 = df["wait_time_min"].quantile(0.25)
    q3 = df["wait_time_min"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df["wait_time_min"] < lower) | (df["wait_time_min"] > upper)]
    negative_waits = df[df["wait_time_min"] < 0]

    profile = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "start timestamp",
                "end timestamp",
                "duplicate rows",
                "unique patients",
                "wait mean",
                "wait median",
                "wait std dev",
                "wait min",
                "wait max",
                "negative waits",
                "IQR lower fence",
                "IQR upper fence",
                "IQR outlier count",
            ],
            "value": [
                f"{df.shape[0]:,}",
                f"{df.shape[1]:,}",
                str(df["timestamp"].min()),
                str(df["timestamp"].max()),
                f"{df.duplicated().sum():,}",
                f"{df['patient_id'].nunique():,}",
                f"{df['wait_time_min'].mean():.2f}",
                f"{df['wait_time_min'].median():.2f}",
                f"{df['wait_time_min'].std():.2f}",
                f"{df['wait_time_min'].min():.2f}",
                f"{df['wait_time_min'].max():.2f}",
                f"{len(negative_waits):,}",
                f"{lower:.2f}",
                f"{upper:.2f}",
                f"{len(outliers):,}",
            ],
        }
    )

    missing = df.isna().sum().to_frame("missing_values")
    ward_raw = df["ward_location"].value_counts().rename_axis("raw_ward").to_frame("rows")
    ward_clean = df["ward_clean"].value_counts().rename_axis("clean_ward").to_frame("rows")
    priority_raw = df["priority"].value_counts().rename_axis("raw_priority").to_frame("rows")
    priority_clean = df["priority_clean"].value_counts().rename_axis("clean_priority").to_frame("rows")
    ward_wait = df.groupby("ward_clean")["wait_time_min"].agg(["count", "mean", "median", "std"]).round(2)
    priority_wait = df.groupby("priority_clean")["wait_time_min"].agg(["count", "mean", "median", "std"]).round(2)
    nurse_wait = df.groupby("nurse_on_duty")["wait_time_min"].agg(["count", "mean", "median", "std"]).round(2)
    hourly = df.groupby("hour")["wait_time_min"].agg(["count", "mean"]).round(2)
    daily_volume = df.groupby("date").size().rename("patients").to_frame()

    return {
        "profile": profile,
        "missing": missing,
        "ward_raw": ward_raw,
        "ward_clean": ward_clean,
        "priority_raw": priority_raw,
        "priority_clean": priority_clean,
        "ward_wait": ward_wait,
        "priority_wait": priority_wait,
        "nurse_wait": nurse_wait,
        "hourly": hourly,
        "daily_volume_head": daily_volume.head(10),
        "outlier_examples": outliers[
            ["timestamp", "patient_id", "ward_location", "priority", "wait_time_min", "nurse_on_duty"]
        ].head(8).round(2),
        "negative_examples": negative_waits[
            ["timestamp", "patient_id", "ward_location", "priority", "wait_time_min", "nurse_on_duty"]
        ].head(8).round(2),
    }


sensor = sensor_analysis()
hospital = hospital_analysis()

cells = [
    md_cell(
        """
# Comprehensive Data Analysis: Sensor Readings and Hospital Wait Times

Prepared in a senior data analyst style, but explained with a light, practical gist. Think of each dataset as its own little mystery: first we inspect the scene, then clean the labels, then look for patterns, oddballs, and useful business takeaways.

**Files analyzed**

- `week2_sensor_readings.csv`
- `hospital_wait_times.csv`

**Notebook flow**

1. Load libraries and settings.
2. Analyze the sensor dataset on its own.
3. Analyze the hospital dataset on its own.
4. Close with practical recommendations.
"""
    ),
    code_cell(
        """
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:,.2f}".format)

DATA_DIR = Path(".")
SENSOR_FILE = DATA_DIR / "week2_sensor_readings.csv"
HOSPITAL_FILE = DATA_DIR / "hospital_wait_times.csv"
"""
    ),
    md_cell(
        """
## Dataset 1: Week 2 Sensor Readings

### 1. Load the data

**Fun gist:** We are opening the control-room logbook. Before we trust it, we peek at the first few rows and ask: "What are we dealing with?"
"""
    ),
    code_cell(
        """
sensor_df = pd.read_csv(SENSOR_FILE)
sensor_df.head()
"""
    ),
    code_cell(
        """
sensor_df.info()
"""
    ),
    md_cell(
        f"""
### 2. Basic profile

**Fun gist:** This is the dataset's passport photo: size, timeline, duplicate count, and the basic pulse of the readings.

{markdown_table(sensor["profile"], index=False)}
"""
    ),
    code_cell(
        """
sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])

sensor_profile = pd.DataFrame({
    "metric": [
        "rows", "columns", "start timestamp", "end timestamp",
        "duplicate rows", "reading mean", "reading median",
        "reading std dev", "reading min", "reading max"
    ],
    "value": [
        sensor_df.shape[0], sensor_df.shape[1],
        sensor_df["timestamp"].min(), sensor_df["timestamp"].max(),
        sensor_df.duplicated().sum(),
        sensor_df["reading"].mean(), sensor_df["reading"].median(),
        sensor_df["reading"].std(), sensor_df["reading"].min(), sensor_df["reading"].max()
    ]
})
sensor_profile
"""
    ),
    md_cell(
        f"""
### 3. Missing values

**Fun gist:** Missing values are like blank boxes on a checklist. Here we count the blanks before making any bold claims.

{markdown_table(sensor["missing"])}
"""
    ),
    code_cell(
        """
sensor_df.isna().sum().to_frame("missing_values")
"""
    ),
    md_cell(
        f"""
### 4. Category consistency check

**Fun gist:** The data is speaking in nicknames: `MBA` and `mombasa` mean Mombasa, while `P1` means `Pressure_01`. We standardize names so the same thing does not wear three hats.

**Raw locations**

{markdown_table(sensor["location_raw"])}

**Cleaned locations**

{markdown_table(sensor["location_clean"])}

**Raw sensors**

{markdown_table(sensor["sensor_raw"])}

**Cleaned sensors**

{markdown_table(sensor["sensor_clean"])}
"""
    ),
    code_cell(
        """
sensor_df["location_clean"] = sensor_df["location"].replace({
    "mombasa": "Mombasa",
    "MBA": "Mombasa",
    "NRB": "Nairobi"
})

sensor_df["sensor_clean"] = sensor_df["sensor_id"].replace({
    "P1": "Pressure_01",
    "TEMP_SENS": "temp_sensor"
})

display(sensor_df["location"].value_counts().to_frame("raw_count"))
display(sensor_df["location_clean"].value_counts().to_frame("clean_count"))
display(sensor_df["sensor_id"].value_counts().to_frame("raw_count"))
display(sensor_df["sensor_clean"].value_counts().to_frame("clean_count"))
"""
    ),
    md_cell(
        f"""
### 5. Reading patterns by shift

**Fun gist:** Now we ask whether readings behave differently by work shift. This is the "does the night shift have a different soundtrack?" question.

{markdown_table(sensor["shift_summary"])}
"""
    ),
    code_cell(
        """
sensor_shift_summary = (
    sensor_df.groupby("shift")["reading"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)
sensor_shift_summary
"""
    ),
    md_cell(
        f"""
### 6. Reading patterns by cleaned location and sensor

**Fun gist:** We now compare the cleaned combinations. This is where messy labels stop splitting the vote.

{markdown_table(sensor["clean_combo"])}
"""
    ),
    code_cell(
        """
sensor_combo_summary = (
    sensor_df.groupby(["location_clean", "sensor_clean"])["reading"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)
sensor_combo_summary
"""
    ),
    md_cell(
        f"""
### 7. Outlier check

**Fun gist:** Outliers are readings standing dramatically in the doorway. Some are real incidents, some are sensor hiccups. We flag them instead of deleting them blindly.

{markdown_table(sensor["outlier_examples"], index=False)}
"""
    ),
    code_cell(
        """
q1 = sensor_df["reading"].quantile(0.25)
q3 = sensor_df["reading"].quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

sensor_outliers = sensor_df[
    (sensor_df["reading"] < lower_fence) |
    (sensor_df["reading"] > upper_fence)
]

print(f"Lower fence: {lower_fence:.2f}")
print(f"Upper fence: {upper_fence:.2f}")
print(f"Outlier rows: {len(sensor_outliers)}")
sensor_outliers.head(8)
"""
    ),
    md_cell(
        """
### 8. Visual exploration

**Fun gist:** Tables tell us the facts; charts help the facts wave at us from across the room.
"""
    ),
    code_cell(
        """
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(sensor_df["reading"], kde=True, ax=axes[0, 0], color="#3a86ff")
axes[0, 0].set_title("Distribution of Sensor Readings")

sns.boxplot(data=sensor_df, x="sensor_clean", y="reading", ax=axes[0, 1])
axes[0, 1].set_title("Readings by Cleaned Sensor")
axes[0, 1].tick_params(axis="x", rotation=20)

sns.countplot(data=sensor_df, x="location_clean", ax=axes[1, 0])
axes[1, 0].set_title("Record Count by Cleaned Location")

sensor_hourly = sensor_df.assign(hour=sensor_df["timestamp"].dt.hour).groupby("hour")["reading"].mean()
sensor_hourly.plot(marker="o", ax=axes[1, 1], color="#ff6b35")
axes[1, 1].set_title("Average Reading by Hour of Day")
axes[1, 1].set_ylabel("Average reading")

plt.tight_layout()
plt.show()
"""
    ),
    md_cell(
        """
### Sensor dataset takeaways

- The file has a complete, regular 5-minute time series with no missing values.
- The biggest data quality issue is inconsistent naming for locations and sensors.
- The readings are centered close to 50, matching the synthetic generation logic.
- Outliers exist, but they should be reviewed rather than automatically removed.
- Cleaned categories make summaries much more reliable because `MBA`, `mombasa`, and `Mombasa` stop being treated as separate places.
"""
    ),
    md_cell(
        """
---

## Dataset 2: Hospital Wait Times

### 1. Load the data

**Fun gist:** This is the hospital queue board. We check who arrived, where they went, how urgent they were, how long they waited, and who was on duty.
"""
    ),
    code_cell(
        """
hospital_df = pd.read_csv(HOSPITAL_FILE)
hospital_df.head()
"""
    ),
    code_cell(
        """
hospital_df.info()
"""
    ),
    md_cell(
        f"""
### 2. Basic profile

**Fun gist:** First we take the hospital dataset's vital signs: rows, time span, patient count, and wait-time behavior.

{markdown_table(hospital["profile"], index=False)}
"""
    ),
    code_cell(
        """
hospital_df["timestamp"] = pd.to_datetime(hospital_df["timestamp"])

hospital_profile = pd.DataFrame({
    "metric": [
        "rows", "columns", "start timestamp", "end timestamp",
        "duplicate rows", "unique patients", "wait mean", "wait median",
        "wait std dev", "wait min", "wait max"
    ],
    "value": [
        hospital_df.shape[0], hospital_df.shape[1],
        hospital_df["timestamp"].min(), hospital_df["timestamp"].max(),
        hospital_df.duplicated().sum(), hospital_df["patient_id"].nunique(),
        hospital_df["wait_time_min"].mean(), hospital_df["wait_time_min"].median(),
        hospital_df["wait_time_min"].std(), hospital_df["wait_time_min"].min(),
        hospital_df["wait_time_min"].max()
    ]
})
hospital_profile
"""
    ),
    md_cell(
        f"""
### 3. Missing values

**Fun gist:** Before drawing conclusions, we check whether any patient records have blank pockets.

{markdown_table(hospital["missing"])}
"""
    ),
    code_cell(
        """
hospital_df.isna().sum().to_frame("missing_values")
"""
    ),
    md_cell(
        f"""
### 4. Category consistency check

**Fun gist:** The hospital data uses aliases too: `ER` and `emergency` are Emergency, `Peds` is Pediatrics, and `GW` is General Ward. Priority also has urgent shouting in two different ways.

**Raw wards**

{markdown_table(hospital["ward_raw"])}

**Cleaned wards**

{markdown_table(hospital["ward_clean"])}

**Raw priorities**

{markdown_table(hospital["priority_raw"])}

**Cleaned priorities**

{markdown_table(hospital["priority_clean"])}
"""
    ),
    code_cell(
        """
hospital_df["ward_clean"] = hospital_df["ward_location"].replace({
    "ER": "Emergency",
    "emergency": "Emergency",
    "Peds": "Pediatrics",
    "GW": "General Ward"
})

hospital_df["priority_clean"] = hospital_df["priority"].replace({
    "urgent": "Urgent",
    "URGENT": "Urgent"
})

display(hospital_df["ward_location"].value_counts().to_frame("raw_count"))
display(hospital_df["ward_clean"].value_counts().to_frame("clean_count"))
display(hospital_df["priority"].value_counts().to_frame("raw_count"))
display(hospital_df["priority_clean"].value_counts().to_frame("clean_count"))
"""
    ),
    md_cell(
        f"""
### 5. Wait time by ward

**Fun gist:** This tells us where the waiting-room traffic feels heaviest. Count shows workload; mean and median show wait experience.

{markdown_table(hospital["ward_wait"])}
"""
    ),
    code_cell(
        """
ward_wait_summary = (
    hospital_df.groupby("ward_clean")["wait_time_min"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)
ward_wait_summary
"""
    ),
    md_cell(
        f"""
### 6. Wait time by priority

**Fun gist:** Priority should ideally shape waiting time. Here we check whether urgent cases are actually moving faster or just wearing a louder label.

{markdown_table(hospital["priority_wait"])}
"""
    ),
    code_cell(
        """
priority_wait_summary = (
    hospital_df.groupby("priority_clean")["wait_time_min"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)
priority_wait_summary
"""
    ),
    md_cell(
        f"""
### 7. Wait time by nurse on duty

**Fun gist:** This is not a blame chart. It is a staffing clue. Differences can come from shift timing, ward mix, patient load, or triage process.

{markdown_table(hospital["nurse_wait"])}
"""
    ),
    code_cell(
        """
nurse_wait_summary = (
    hospital_df.groupby("nurse_on_duty")["wait_time_min"]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)
nurse_wait_summary
"""
    ),
    md_cell(
        f"""
### 8. Negative and unusual wait times

**Fun gist:** A negative wait time means the clock has started telling jokes. In real operations, these rows should be corrected or excluded from service metrics after investigation.

**Negative wait examples**

{markdown_table(hospital["negative_examples"], index=False)}

**IQR outlier examples**

{markdown_table(hospital["outlier_examples"], index=False)}
"""
    ),
    code_cell(
        """
q1 = hospital_df["wait_time_min"].quantile(0.25)
q3 = hospital_df["wait_time_min"].quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

negative_waits = hospital_df[hospital_df["wait_time_min"] < 0]
hospital_outliers = hospital_df[
    (hospital_df["wait_time_min"] < lower_fence) |
    (hospital_df["wait_time_min"] > upper_fence)
]

print(f"Negative wait rows: {len(negative_waits)}")
print(f"Lower fence: {lower_fence:.2f}")
print(f"Upper fence: {upper_fence:.2f}")
print(f"Outlier rows: {len(hospital_outliers)}")

display(negative_waits.head(8))
display(hospital_outliers.head(8))
"""
    ),
    md_cell(
        """
### 9. Visual exploration

**Fun gist:** Time to turn the waiting-room ledger into pictures: distribution, ward pressure, priority comparison, and nurse workload.
"""
    ),
    code_cell(
        """
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.histplot(hospital_df["wait_time_min"], kde=True, ax=axes[0, 0], color="#2a9d8f")
axes[0, 0].set_title("Distribution of Wait Times")

sns.boxplot(data=hospital_df, x="ward_clean", y="wait_time_min", ax=axes[0, 1])
axes[0, 1].set_title("Wait Time by Cleaned Ward")
axes[0, 1].tick_params(axis="x", rotation=20)

sns.boxplot(data=hospital_df, x="priority_clean", y="wait_time_min", ax=axes[1, 0])
axes[1, 0].set_title("Wait Time by Cleaned Priority")

sns.countplot(data=hospital_df, x="nurse_on_duty", ax=axes[1, 1])
axes[1, 1].set_title("Patient Records by Nurse on Duty")
axes[1, 1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.show()
"""
    ),
    code_cell(
        """
hospital_df["date"] = hospital_df["timestamp"].dt.date
hospital_df["hour"] = hospital_df["timestamp"].dt.hour

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

daily_volume = hospital_df.groupby("date").size()
daily_volume.plot(marker="o", ax=axes[0], color="#7b2cbf")
axes[0].set_title("Daily Patient Record Volume")
axes[0].set_ylabel("Patients")

hourly_wait = hospital_df.groupby("hour")["wait_time_min"].mean()
hourly_wait.plot(marker="o", ax=axes[1], color="#f77f00")
axes[1].set_title("Average Wait Time by Hour")
axes[1].set_ylabel("Average wait time, minutes")

plt.tight_layout()
plt.show()
"""
    ),
    md_cell(
        """
### Hospital dataset takeaways

- The file has no missing values, but category standardization is essential.
- Emergency, ER, and emergency should be analyzed as one ward.
- Urgent and URGENT should be normalized before any priority reporting.
- Negative wait times are operationally impossible and should be investigated.
- Nurse-level comparisons are useful, but they need context before being interpreted as performance differences.
- The data is synthetic, so patterns may be evenly distributed; still, the workflow mirrors real hospital analytics.
"""
    ),
    md_cell(
        """
---

## Practical Recommendations

### For the sensor dataset

1. Create official lookup tables for `sensor_id` and `location`.
2. Keep raw labels, but add cleaned reporting columns.
3. Flag outliers for review instead of deleting them immediately.
4. Monitor readings by sensor, shift, and location after standardization.

### For the hospital dataset

1. Standardize ward and priority names at data entry.
2. Treat negative wait times as data quality exceptions.
3. Track wait time by cleaned ward, priority, hour, and nurse assignment.
4. Use nurse comparisons as a starting point for workflow review, not as a standalone performance score.

**Final analyst gist:** Clean labels first, summarize second, visualize third, and only then make decisions. Messy categories can make a calm dataset look chaotic.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(NOTEBOOK_PATH)
