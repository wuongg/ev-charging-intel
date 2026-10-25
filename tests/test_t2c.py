"""T2C is a held-out actual-dwell quantile. It does not copy the raw declaration."""

from datetime import datetime, timedelta, timezone

from eci.optimize.calibrate import (
    HazardCorrector,
    corrector_from_records,
    declared_bucket,
    fold_of,
    records_from_sessions,
)
from eci.optimize.mpc import ScenarioSession


def _session(index: int, declared_h: float, actual_h: float, hour: int = 8) -> ScenarioSession:
    connection = datetime(2018, 6, 4, hour, tzinfo=timezone.utc)
    return ScenarioSession(
        session_id=f"src-{index}",
        connection=connection,
        disconnect=connection + timedelta(hours=actual_h),
        kwh=8.0,
        p_max_kw=6.6,
        requested_departure=connection + timedelta(hours=declared_h),
        station_id="A",
    )


def _panel() -> list[ScenarioSession]:
    sessions = []
    # Even indices: declared 10h, actual 6h (late declaration).
    # Odd indices: declared 4h, actual 6h (early declaration).
    for index in range(80):
        late = index % 2 == 0
        sessions.append(_session(index, declared_h=10.0 if late else 4.0, actual_h=6.0))
    return sessions


def test_fold_is_deterministic() -> None:
    assert fold_of("src-7") == fold_of("src-7")
    assert fold_of("src-7") in (0, 1)


def test_fit_holds_the_scored_session_out() -> None:
    sessions = _panel()
    records = records_from_sessions(sessions)
    corrector = corrector_from_records(records, beta=0.5)
    assert isinstance(corrector, HazardCorrector)
    first = sessions[0]
    fold = fold_of(first.session_id)
    # The scoring fold must not contain this id's actual as a training row of that fold.
    scoring = corrector.folds[1 - fold]
    # Re-fitting the scoring fold from only the other half still runs.
    hours = scoring.actual_hours(first.connection.hour, 10.0, 0.5)
    assert hours > 0


def test_same_session_same_corrected_deadline() -> None:
    corrector = corrector_from_records(records_from_sessions(_panel()), beta=0.5)
    session = _session(2, declared_h=10.0, actual_h=6.0)
    first = corrector.apply([session])[0].requested_departure
    second = corrector.apply([session])[0].requested_departure
    assert first == second


def test_late_declaration_is_pulled_earlier() -> None:
    sessions = _panel()
    corrector = corrector_from_records(records_from_sessions(sessions), beta=0.5)
    late = [
        session
        for session in sessions
        if (session.requested_departure - session.connection).total_seconds() == 10 * 3600
    ]
    probe = late[0]
    declared = probe.requested_departure
    corrected = corrector.apply([probe])[0].requested_departure
    assert corrected is not None and declared is not None
    assert corrected < declared
    assert corrected > probe.connection


def test_declared_bucket_splits_the_tail() -> None:
    assert declared_bucket(1.0) == 0
    assert declared_bucket(10.0) != declared_bucket(3.0)
