"""Slot length, column names, quality flags, and invariant identifiers.

Invariant *checks* are not implemented. The enum is the contract chaos and the
feasibility checker will share later.
"""

from __future__ import annotations

from enum import Enum

SLOT_MINUTES = 15

# Audit thresholds from the day-1 data contract. Not tariff prices.
LONG_DWELL_HOURS = 48
LONG_DWELL_REPORT_HOURS = 24
KWH_MIN = 0.0
KWH_MAX = 120.0
UI_MINUTES_ABSURD = 5
UI_WH_PER_MILE_MAX = 500
CLOCK_SKEW_RATE_MIN = 0.02
CLOCK_SKEW_RATE_MAX = 0.10
ARRIVAL_PEAK_HOURS = (7, 8, 9, 10)
REFERENCE_RELATIVE_TOLERANCE = 0.20


class COLS(str, Enum):
    """Column names used past the audit gate."""

    ID = "_id"
    CLUSTER_ID = "clusterID"
    CONNECTION_TIME = "connectionTime"
    DISCONNECT_TIME = "disconnectTime"
    DONE_CHARGING_TIME = "doneChargingTime"
    KWH = "kWhDelivered"
    SESSION_ID = "sessionID"
    SITE_ID = "siteID"
    SPACE_ID = "spaceID"
    STATION_ID = "stationID"
    TIMEZONE = "timezone"
    USER_ID = "userID"
    USER_INPUTS = "userInputs"
    USER_INPUTS_LAST = "userInputs_last"
    CONNECTION_UTC = "connectionTime_utc"
    CONNECTION_LOCAL = "connectionTime_local"
    DISCONNECT_UTC = "disconnectTime_utc"
    DISCONNECT_LOCAL = "disconnectTime_local"
    DONE_CHARGING_UTC = "doneChargingTime_utc"
    DONE_CHARGING_LOCAL = "doneChargingTime_local"
    N_USER_INPUTS = "n_user_inputs"
    EXTRA_JSON = "extra_json"


F_CLOCK_SKEW = "f_clock_skew"
F_LONG_DWELL = "f_long_dwell"
F_UI_ABSURD = "f_ui_absurd"
F_DST_CROSS = "f_dst_cross"
F_TYPE_COERCED = "f_type_coerced"
F_KWH_OOR = "f_kwh_oor"

FLAGS = (
    F_CLOCK_SKEW,
    F_LONG_DWELL,
    F_UI_ABSURD,
    F_DST_CROSS,
    F_TYPE_COERCED,
    F_KWH_OOR,
)


class Invariant(str, Enum):
    """Scheduler invariants. Checks are intentionally not implemented on day 1.

    I1 site cap: metered site power never exceeds the configured site cap.
    I2 EVSE bounds: each port's power stays inside [0, pmax].
    I3 energy balance: delivered + unmet equals required energy (soft SLA).
    I4 window: no energy is delivered outside the connection interval.
    I5 nonnegative: power, energy, and unmet are nonnegative.
    I6 peak epigraph: the reported peak is at least the maximum slot power.
    I7 causality: a decision at t uses only information available at t
        (userInputs snapshot at or before connection; never doneChargingTime).
    I8 independent check: the feasibility checker agrees on every emitted schedule.
    I9 plant-actual separation: committed power is gated by the physical plug
        (connection, disconnect), never by a declared or imputed deadline.
    """

    I1_SITE_CAP = "I1"
    I2_EVSE_BOUNDS = "I2"
    I3_ENERGY_BALANCE = "I3"
    I4_WINDOW = "I4"
    I5_NONNEGATIVE = "I5"
    I6_PEAK_EPIGRAPH = "I6"
    I7_CAUSALITY = "I7"
    I8_INDEPENDENT_CHECK = "I8"
    I9_PLANT_ACTUAL = "I9"
