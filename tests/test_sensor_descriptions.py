"""Sanity tests for the C-MAPSS sensor descriptions module."""

import sys
sys.path.insert(0, ".")

from src.sensor_descriptions import (
    SENSOR_TABLE,
    INTERPRETATION,
    describe_sensor,
    subsystem_of,
)


def test_sensor_table_has_all_21_sensors():
    assert len(SENSOR_TABLE) == 21
    assert list(SENSOR_TABLE.columns) == [
        "symbol", "quantity", "units", "subsystem"
    ]


def test_describe_sensor_format():
    assert describe_sensor("sensor_11") == (
        "Sensor 11 (Ps30 — HPC outlet static pressure, psia, HPC)"
    )
    assert describe_sensor("sensor_3") == (
        "Sensor 3 (T30 — HPC outlet total temperature, °R, HPC)"
    )


def test_describe_sensor_unknown_passthrough():
    assert describe_sensor("not_a_sensor") == "not_a_sensor"


def test_subsystem_of_known_grouping():
    assert subsystem_of("sensor_3") == "HPC"
    assert subsystem_of("sensor_4") == "LPT"
    assert subsystem_of("sensor_8") == "Fan"
    assert subsystem_of("not_a_sensor") == "Unknown"


def test_interpretation_covers_informative_sensors():
    # Every sensor kept in FD001 (after constant removal) should have an
    # interpretation paragraph available.
    fd001_kept = {
        "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
        "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
        "sensor_15", "sensor_17", "sensor_20", "sensor_21",
    }
    missing = fd001_kept - set(INTERPRETATION.keys())
    assert not missing, f"Missing interpretation for: {missing}"
