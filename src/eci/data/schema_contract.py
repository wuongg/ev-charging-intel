"""Session schema: 13 ACN keys, explicit dtypes, siteID zero-padding."""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

# Order matches a raw ACN session document.
SESSION_KEYS: tuple[str, ...] = (
    "_id",
    "clusterID",
    "connectionTime",
    "disconnectTime",
    "doneChargingTime",
    "kWhDelivered",
    "sessionID",
    "siteID",
    "spaceID",
    "stationID",
    "timezone",
    "userID",
    "userInputs",
)

if len(SESSION_KEYS) != 13:
    raise RuntimeError("ACN session schema is defined as 13 keys")

# Logical types for the dictionary. Parquet stores userInputs as JSON text
# because a list-of-dicts does not have a stable struct.
DTYPE_TABLE: dict[str, dict[str, str]] = {
    "_id": {"dtype": "string", "unit": "id", "description": "ACN document id"},
    "clusterID": {"dtype": "string", "unit": "id", "description": "Zero-padded cluster id"},
    "connectionTime": {
        "dtype": "string",
        "unit": "RFC1123 GMT",
        "description": "Original connection timestamp",
    },
    "disconnectTime": {
        "dtype": "string",
        "unit": "RFC1123 GMT",
        "description": "Original disconnect timestamp",
    },
    "doneChargingTime": {
        "dtype": "string",
        "unit": "RFC1123 GMT",
        "description": "Original done-charging timestamp; null when absent",
    },
    "kWhDelivered": {
        "dtype": "float64",
        "unit": "kWh",
        "description": "Energy delivered over the session",
    },
    "sessionID": {"dtype": "string", "unit": "id", "description": "ACN session id"},
    "siteID": {
        "dtype": "string",
        "unit": "id",
        "description": "Zero-padded 4-character site id. Integer 2 becomes '0002'.",
    },
    "spaceID": {"dtype": "string", "unit": "id", "description": "Parking space id"},
    "stationID": {"dtype": "string", "unit": "id", "description": "EVSE / station id"},
    "timezone": {"dtype": "string", "unit": "IANA", "description": "Site timezone name on the row"},
    "userID": {
        "dtype": "string",
        "unit": "id",
        "description": "Driver id. Null is common on the Caltech mirror.",
    },
    "userInputs": {
        "dtype": "json list-of-dicts",
        "unit": "events",
        "description": "Full userInputs history, preserved as a JSON list of dicts",
    },
    "userInputs_last": {
        "dtype": "json dict",
        "unit": "snapshot",
        "description": (
            "Decision-time snapshot. Latest createdOn (else modifiedAt) at or before "
            "connectionTime. If every stamp is after plug-in, the earliest input — "
            "later edits are post-hoc."
        ),
    },
    "connectionTime_utc": {
        "dtype": "datetime64[us, UTC]",
        "unit": "UTC",
        "description": "Connection instant",
    },
    "connectionTime_local": {
        "dtype": "datetime64[us, America/Los_Angeles]",
        "unit": "local",
        "description": "Connection wall time in the site timezone",
    },
    "disconnectTime_utc": {
        "dtype": "datetime64[us, UTC]",
        "unit": "UTC",
        "description": "Disconnect instant",
    },
    "disconnectTime_local": {
        "dtype": "datetime64[us, America/Los_Angeles]",
        "unit": "local",
        "description": "Disconnect wall time in the site timezone",
    },
    "doneChargingTime_utc": {
        "dtype": "datetime64[us, UTC]",
        "unit": "UTC",
        "description": "Done-charging instant",
    },
    "doneChargingTime_local": {
        "dtype": "datetime64[us, America/Los_Angeles]",
        "unit": "local",
        "description": "Done-charging wall time in the site timezone",
    },
    "n_user_inputs": {
        "dtype": "int64",
        "unit": "count",
        "description": "Length of the original userInputs array (0 when null)",
    },
    "extra_json": {
        "dtype": "json",
        "unit": "",
        "description": "Unknown session keys, kept and logged",
    },
    "f_clock_skew": {
        "dtype": "bool",
        "unit": "flag",
        "description": "doneChargingTime > disconnectTime. Kept; drop only in completion-time studies.",
    },
    "f_long_dwell": {
        "dtype": "bool",
        "unit": "flag",
        "description": "UTC dwell longer than 48 hours. Kept; long parking is real occupancy.",
    },
    "f_ui_absurd": {
        "dtype": "bool",
        "unit": "flag",
        "description": "At least one userInputs entry failed the absurd-input rule and was dropped.",
    },
    "f_dst_cross": {
        "dtype": "bool",
        "unit": "flag",
        "description": "UTC duration differs from the local wall-clock duration (DST boundary).",
    },
    "f_type_coerced": {
        "dtype": "bool",
        "unit": "flag",
        "description": "A field was coerced (siteID/clusterID padding, non-string id, userInputs dict).",
    },
    "f_kwh_oor": {
        "dtype": "bool",
        "unit": "flag",
        "description": "kWhDelivered outside [0, 120]. Kept.",
    },
}

_SEEN_EXTRA: set[str] = set()


def assert_session_keys(record: dict) -> None:
    missing = [key for key in SESSION_KEYS if key not in record]
    if missing:
        raise AssertionError(f"session missing keys: {missing}")


def coerce_padded_id(value: object) -> tuple[str | None, bool]:
    """Return (zero-padded id, coerced?). '0002' is unchanged; 2 and '2' become '0002'."""
    if value is None:
        return None, False
    if isinstance(value, bool):
        return str(value), True
    if isinstance(value, int):
        return f"{value:04d}", True
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):04d}", True
    text = str(value).strip()
    if text.isdigit():
        padded = text.zfill(4)
        changed = padded != text or not isinstance(value, str)
        return padded, changed
    if not isinstance(value, str):
        return text, True
    return text, False


def _as_string(value: object) -> tuple[str | None, bool]:
    if value is None:
        return None, False
    if isinstance(value, str):
        return value, False
    return str(value), True


def _as_float(value: object) -> tuple[float | None, bool]:
    if value is None or value == "":
        return None, False
    if isinstance(value, bool):
        return float(int(value)), True
    if isinstance(value, (int, float)):
        return float(value), False
    try:
        return float(str(value)), True
    except ValueError:
        return None, True


def coerce_session(raw: dict) -> tuple[dict, bool]:
    """Copy a raw session onto the 13-key contract.

    Unknown keys are logged once per name and kept under ``extra``.
    """
    if not isinstance(raw, dict):
        raise TypeError("session must be a dict")
    extra_keys = sorted(set(raw) - set(SESSION_KEYS))
    for key in extra_keys:
        if key not in _SEEN_EXTRA:
            _SEEN_EXTRA.add(key)
            LOGGER.info("unknown extra session key kept: %s", key)

    coerced = False
    out: dict = {}
    site_id, site_changed = coerce_padded_id(raw.get("siteID"))
    cluster_id, cluster_changed = coerce_padded_id(raw.get("clusterID"))
    coerced = coerced or site_changed or cluster_changed
    out["siteID"] = site_id
    out["clusterID"] = cluster_id

    string_keys = (
        "_id",
        "connectionTime",
        "disconnectTime",
        "doneChargingTime",
        "sessionID",
        "spaceID",
        "stationID",
        "timezone",
    )
    for key in string_keys:
        text, changed = _as_string(raw.get(key))
        out[key] = text
        coerced = coerced or changed

    user_id, user_changed = _as_string(raw.get("userID"))
    out["userID"] = user_id
    coerced = coerced or user_changed

    kwh, kwh_changed = _as_float(raw.get("kWhDelivered"))
    out["kWhDelivered"] = kwh
    coerced = coerced or kwh_changed

    inputs = raw.get("userInputs")
    if inputs is None:
        out["userInputs"] = None
    elif isinstance(inputs, list):
        out["userInputs"] = inputs
    elif isinstance(inputs, dict):
        out["userInputs"] = [inputs]
        coerced = True
        LOGGER.info("userInputs dict coerced to a one-element list")
    else:
        out["userInputs"] = None
        coerced = True
        LOGGER.info("userInputs of type %s dropped to null", type(inputs).__name__)

    out["extra"] = {key: raw[key] for key in extra_keys} if extra_keys else None
    assert_session_keys(out)
    return out, coerced
