"""
NASA C-MAPSS sensor descriptions.

Maps the 21 raw sensors from the C-MAPSS dataset to their physical
meanings (per Saxena et al., PHM08), an engine subsystem grouping, and
a one-line interpretation guide. Used by:

- ``notebooks/07_feature_importance.ipynb`` for mechanistic
  interpretation of SHAP rankings.
- ``app/streamlit_app.py`` for richer tooltips in the SHAP panel.

The subsystem grouping is the lens that turns a flat top-N feature
list into an interpretable story ("the HPC sensors dominate" is more
useful than "sensors 3, 7, 11, 17 dominate").
"""

from __future__ import annotations

import pandas as pd

# (sensor_name, short_label, physical_quantity, units, subsystem)
_SENSORS: list[tuple[str, str, str, str, str]] = [
    ("sensor_1",  "T2",          "Fan inlet total temperature",     "°R",      "Fan"),
    ("sensor_2",  "T24",         "LPC outlet total temperature",    "°R",      "LPC"),
    ("sensor_3",  "T30",         "HPC outlet total temperature",    "°R",      "HPC"),
    ("sensor_4",  "T50",         "LPT outlet total temperature",    "°R",      "LPT"),
    ("sensor_5",  "P2",          "Fan inlet pressure",              "psia",    "Fan"),
    ("sensor_6",  "P15",         "Bypass-duct total pressure",      "psia",    "Fan"),
    ("sensor_7",  "P30",         "HPC outlet total pressure",       "psia",    "HPC"),
    ("sensor_8",  "Nf",          "Physical fan speed",              "rpm",     "Fan"),
    ("sensor_9",  "Nc",          "Physical core speed",             "rpm",     "Core"),
    ("sensor_10", "epr",         "Engine pressure ratio (P50/P2)",  "—",       "Performance"),
    ("sensor_11", "Ps30",        "HPC outlet static pressure",      "psia",    "HPC"),
    ("sensor_12", "phi",         "Fuel-flow / Ps30 ratio",          "pps/psia", "Combustor"),
    ("sensor_13", "NRf",         "Corrected fan speed",             "rpm",     "Fan"),
    ("sensor_14", "NRc",         "Corrected core speed",            "rpm",     "Core"),
    ("sensor_15", "BPR",         "Bypass ratio",                    "—",       "Performance"),
    ("sensor_16", "farB",        "Burner fuel-air ratio",           "—",       "Combustor"),
    ("sensor_17", "htBleed",     "Bleed enthalpy",                  "—",       "HPC"),
    ("sensor_18", "Nf_dmd",      "Demanded fan speed",              "rpm",     "Control"),
    ("sensor_19", "PCNfR_dmd",   "Demanded corrected fan speed",    "rpm",     "Control"),
    ("sensor_20", "W31",         "HPT coolant bleed",               "lbm/s",   "HPT"),
    ("sensor_21", "W32",         "LPT coolant bleed",               "lbm/s",   "LPT"),
]


SENSOR_TABLE: pd.DataFrame = pd.DataFrame(
    _SENSORS,
    columns=["sensor", "symbol", "quantity", "units", "subsystem"],
).set_index("sensor")


# What a rising / falling reading in this sensor usually means for the
# engine. Used as one-line interpretation for top-feature paragraphs.
INTERPRETATION: dict[str, str] = {
    "sensor_2": (
        "LPC outlet temperature. Rises when the low-pressure compressor's "
        "efficiency drops — early indicator of upstream wear."
    ),
    "sensor_6": (
        "Bypass-duct total pressure. Shifts indirectly as the fan/core "
        "balance changes when downstream components (HPC) degrade — picks "
        "up airflow redistribution that direct HPC sensors don't see."
    ),
    "sensor_3": (
        "HPC outlet temperature. The classic HPC-degradation signal — rises "
        "as compressor blades wear and inlet conditions deteriorate."
    ),
    "sensor_4": (
        "LPT outlet temperature. Reflects how much energy reaches the "
        "low-pressure turbine; drifts late as the whole gas path shifts."
    ),
    "sensor_7": (
        "HPC outlet total pressure. Falls as the high-pressure compressor "
        "loses compression efficiency — direct degradation signal."
    ),
    "sensor_8": (
        "Physical fan speed. Mostly controlled, so drift here points to "
        "the controller compensating for downstream losses."
    ),
    "sensor_9": (
        "Physical core speed. The core speeds up as the HPC degrades to "
        "maintain target thrust — strong HPC-degradation indicator."
    ),
    "sensor_11": (
        "HPC outlet static pressure. Falls with HPC degradation; volatility "
        "and kurtosis of this signal rise notably before mean drift."
    ),
    "sensor_12": (
        "Fuel-flow / Ps30. Climbs as the engine needs more fuel for the "
        "same thrust — indirect efficiency-loss signal."
    ),
    "sensor_13": (
        "Corrected fan speed. Operating-condition-normalised fan speed; "
        "useful because raw RPM mixes regime + degradation."
    ),
    "sensor_14": (
        "Corrected core speed. Like NRf but for the core; close kin to "
        "sensor 9 with regime correction."
    ),
    "sensor_15": (
        "Bypass ratio. Shifts as fan and core balance changes during "
        "degradation; tends to drift very late."
    ),
    "sensor_17": (
        "Bleed enthalpy. Energy carried in bleed air — climbs as HPC "
        "outlet temperatures rise."
    ),
    "sensor_20": (
        "HPT coolant bleed. The controller increases it as the high-"
        "pressure turbine runs hotter due to upstream degradation."
    ),
    "sensor_21": (
        "LPT coolant bleed. Same idea as W31 but for the low-pressure "
        "turbine; correlated with W31 in failure mode."
    ),
}


def describe_sensor(sensor_name: str) -> str:
    """Return ``"Sensor N (symbol — quantity, units, subsystem)"`` or a
    fallback if the sensor name isn't in the C-MAPSS spec."""
    if sensor_name not in SENSOR_TABLE.index:
        return sensor_name
    row = SENSOR_TABLE.loc[sensor_name]
    n = sensor_name.replace("sensor_", "")
    return f"Sensor {n} ({row['symbol']} — {row['quantity']}, {row['units']}, {row['subsystem']})"


def subsystem_of(sensor_name: str) -> str:
    """Return the engine subsystem for a sensor, or 'Unknown' if missing."""
    if sensor_name not in SENSOR_TABLE.index:
        return "Unknown"
    return str(SENSOR_TABLE.loc[sensor_name, "subsystem"])
