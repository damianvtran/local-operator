"""The refusal tables: every code, the fallback, the 409, and ``Retry-After``.

WHY THESE ARE PURE-FUNCTION TESTS. The client's tests prove the plumbing — a
response becomes one typed error — while these pin the DECISIONS the tables
encode, one row per contract code. A code the contract gains that this build
does not map fails here rather than surfacing to an operator as a generic
failure, so the test is the review of the table rather than a sample of it.
"""

from __future__ import annotations

import datetime
import email.utils

import pytest

from local_operator.remote_compute.errors import (
    ATTESTATION_CODE_PREFIX,
    ERROR_BY_CODE,
    ERROR_BY_STATUS,
    LOGIN_REQUIRED,
    REFUSAL_CODES,
    CapacityUnavailable,
    Conflict,
    IdempotencyConflict,
    InsufficientCredit,
    InvalidRequest,
    PreviewNotEnabled,
    RemoteComputeError,
    RunBusy,
    RunLimitReached,
    RunNotFound,
    refusal_code,
    refusal_for,
    refusal_message,
    retry_after_seconds,
)
from local_operator.tunnels.errors import LoginRequired

#: Code → the class an operator's remedy is named by, as the contract defines it.
EXPECTED_BY_CODE: dict[str, type[RemoteComputeError] | object] = {
    "preview_not_enabled": PreviewNotEnabled,
    "insufficient_credits": InsufficientCredit,
    "run_limit": RunLimitReached,
    "workload_limit": RunLimitReached,
    "idempotency_conflict": IdempotencyConflict,
    "run_busy": RunBusy,
    "workload_busy": RunBusy,
    "name_claimed": Conflict,
    "not_found": RunNotFound,
    "run_finished": Conflict,
    "revoked": LOGIN_REQUIRED,
    "invalid_request": InvalidRequest,
    "invalid_idempotency_key": InvalidRequest,
    "capacity_unavailable": CapacityUnavailable,
    "admission_unavailable": CapacityUnavailable,
    "not_configured": Conflict,
}


def test_every_contract_code_is_mapped_and_every_mapped_code_is_a_contract_code() -> None:
    """The table and the accepted-body set are the same set, both directions.

    A code accepted from a body but absent from the table would fall through to
    the status (a silent regression); a code in the table but not accepted from a
    body could never fire.
    """
    assert set(ERROR_BY_CODE) == REFUSAL_CODES
    assert set(EXPECTED_BY_CODE) == REFUSAL_CODES


@pytest.mark.parametrize(("code", "expected"), sorted(EXPECTED_BY_CODE.items()))
def test_each_code_maps_to_its_class(code: str, expected: type[Exception] | object) -> None:
    error = refusal_for(409, code=code)
    if expected is LOGIN_REQUIRED:
        assert isinstance(error, LoginRequired)
    else:
        assert type(error) is expected
    # The code is carried on the RemoteComputeError either way, so a surface can
    # be more precise than the class it shares with a sibling code.
    if isinstance(error, RemoteComputeError):
        assert error.code == code
        assert error.status == 409


@pytest.mark.parametrize(
    "code",
    sorted(c for c in EXPECTED_BY_CODE if c not in {"revoked"}),
)
def test_a_coded_refusal_does_not_depend_on_its_status(code: str) -> None:
    """The STATUS cannot change the reading once a code is present."""
    by_conflict = refusal_for(409, code=code)
    by_bad_request = refusal_for(400, code=code)
    assert type(by_conflict) is type(by_bad_request)


def test_a_codeless_refusal_falls_back_to_the_status() -> None:
    """Today's servers send no code; their behaviour is unchanged but for 409."""
    assert isinstance(refusal_for(400), InvalidRequest)
    assert isinstance(refusal_for(402), InsufficientCredit)
    assert isinstance(refusal_for(403), PreviewNotEnabled)
    assert isinstance(refusal_for(404), RunNotFound)
    assert isinstance(refusal_for(503), CapacityUnavailable)
    assert isinstance(refusal_for(401), LoginRequired)


def test_an_unlisted_status_is_the_base_class_and_not_a_guess() -> None:
    error = refusal_for(500)
    assert type(error) is RemoteComputeError
    assert error.status == 500


def test_a_codeless_409_is_generic_rather_than_an_idempotency_conflict() -> None:
    assert type(refusal_for(409)) is Conflict


def test_a_codeless_409_naming_idempotency_is_a_conflict_of_that_kind() -> None:
    message = "This idempotency key was already used with a different request body."
    assert type(refusal_for(409, message=message)) is IdempotencyConflict


@pytest.mark.parametrize(
    "message",
    [
        "You have reached your concurrent run limit.",
        "The workload is busy.",
        "This name is already claimed by another run.",
        "The run has already finished.",
        "idempotency",
    ],
)
def test_a_codeless_409_naming_something_else_stays_generic(message: str) -> None:
    """The message is consulted for ONE reading, and only when it is precise."""
    assert type(refusal_for(409, message=message)) is Conflict


def test_an_attestation_code_is_a_conflict_this_build_cannot_remedy() -> None:
    error = refusal_for(409, code=ATTESTATION_CODE_PREFIX + "nonce")
    assert type(error) is Conflict
    assert isinstance(error, RemoteComputeError)
    assert error.code == ATTESTATION_CODE_PREFIX + "nonce"


def test_an_unknown_code_is_ignored_and_the_status_answers() -> None:
    assert type(refusal_for(403, code="some_future_code")) is PreviewNotEnabled


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"code": "run_limit"}, "run_limit"),
        ({"error": {"code": "run_busy"}}, "run_busy"),
        ({"detail": {"code": "not_found"}}, "not_found"),
        ({"code": "attestation_image"}, "attestation_image"),
        ({"code": "not_a_code"}, None),
        ({"code": ""}, None),
        ({"code": 7}, None),
        ({"error": "plain text"}, None),
        ({"msg": "no code here"}, None),
        (None, None),
        ([], None),
        ("body", None),
    ],
)
def test_only_a_contract_code_is_read_out_of_a_body(body: object, expected: str | None) -> None:
    """A body can carry one enum member into an error, and nothing else."""
    assert refusal_code(body) == expected


@pytest.mark.parametrize(
    "body",
    [
        {"msg": "the sentence"},
        {"message": "the sentence"},
        {"error": "the sentence"},
        {"detail": "the sentence"},
        {"msg": ""},
        {"msg": 3},
        {},
        None,
    ],
)
def test_the_disambiguation_sentence_is_read_or_absent(body: object) -> None:
    sentence = refusal_message(body)
    assert sentence is None or sentence == "the sentence"


def test_a_long_message_is_capped_before_it_is_kept() -> None:
    assert refusal_message({"msg": "x" * 5000}) == "x" * 500


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("0", 0.0),
        ("17", 17.0),
        ("17.5", None),
        ("-5", None),
        ("Wed, 21 Oct 2015 07:28:00 GMT", 0.0),
        ("nonsense", None),
    ],
)
def test_retry_after_parsing_that_does_not_need_a_clock(
    value: str | None, expected: float | None
) -> None:
    assert retry_after_seconds(value) == expected


def test_retry_after_as_a_date_is_measured_from_now() -> None:
    now = datetime.datetime(2026, 11, 17, 12, 0, 0, tzinfo=datetime.timezone.utc)
    later = now + datetime.timedelta(seconds=120)
    header = email.utils.format_datetime(later, usegmt=True)
    assert retry_after_seconds(header, now=now) == pytest.approx(120.0, abs=1)


def test_retry_after_in_the_past_is_zero_rather_than_negative() -> None:
    now = datetime.datetime(2026, 11, 17, 12, 0, 0, tzinfo=datetime.timezone.utc)
    earlier = now - datetime.timedelta(hours=1)
    header = email.utils.format_datetime(earlier, usegmt=True)
    assert retry_after_seconds(header, now=now) == 0.0


def test_retry_after_absent_from_the_error_when_the_header_is_absent() -> None:
    plain = refusal_for(503)
    assert isinstance(plain, RemoteComputeError)
    assert plain.retry_after is None
    delayed = refusal_for(503, retry_after=5.0)
    assert isinstance(delayed, RemoteComputeError)
    assert delayed.retry_after == 5.0


def test_the_status_table_names_only_statuses_the_contract_has() -> None:
    """A row for a status the contract never sends would be dead code."""
    assert set(ERROR_BY_STATUS) <= {400, 401, 402, 403, 404, 409, 429, 503}
    assert ERROR_BY_STATUS[409] is Conflict
