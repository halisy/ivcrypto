from __future__ import annotations

import http.client
import json
import random
import ssl
import urllib.error
from pathlib import Path

import pytest

from ivcrypto.data.client import (
    TOO_MANY_REQUESTS,
    DeribitAPIError,
    DeribitClient,
    DeribitRequestError,
    HttpResponse,
    RateLimit,
    RetryPolicy,
    TokenBucket,
)

DERIBIT_FIXTURES = Path(__file__).parent / "fixtures" / "deribit"


class FakeClock:
    """Deterministic clock: ``sleep`` advances time instantly and is recorded."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedTransport:
    """Returns or raises the scripted outcomes in order and records the URLs."""

    def __init__(self, *outcomes: HttpResponse | BaseException) -> None:
        self.outcomes = list(outcomes)
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> HttpResponse:
        self.urls.append(url)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def fixture_response(name: str, status: int = 200) -> HttpResponse:
    return HttpResponse(status, (DERIBIT_FIXTURES / name).read_bytes())


def index_ok() -> HttpResponse:
    return fixture_response("get_index_price_btc_usd.json")


def rate_limit_error(status: int) -> HttpResponse:
    # Synthetic: Deribit documents code 10028 but it cannot be triggered politely.
    body = {"jsonrpc": "2.0", "error": {"code": TOO_MANY_REQUESTS, "message": "too_many_requests"}}
    return HttpResponse(status, json.dumps(body).encode())


def make_client(transport: ScriptedTransport, **kwargs: object) -> tuple[DeribitClient, FakeClock]:
    clock = FakeClock()
    client = DeribitClient(
        transport=transport, clock=clock.time, sleep=clock.sleep, seed=0, **kwargs
    )
    return client, clock


def test_success_returns_result_and_server_times():
    transport = ScriptedTransport(index_ok())
    client, clock = make_client(transport)
    result = client.call("public/get_index_price", index_name="btc_usd")
    payload = json.loads((DERIBIT_FIXTURES / "get_index_price_btc_usd.json").read_text())
    assert result.result == payload["result"]
    assert result.us_in == payload["usIn"]
    assert result.us_out == payload["usOut"]
    assert result.testnet is False
    assert result.method == "public/get_index_price"
    assert result.params == {"index_name": "btc_usd"}
    assert transport.urls == [
        "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"
    ]
    assert clock.sleeps == []


def test_url_encodes_booleans_in_lowercase():
    client, _ = make_client(ScriptedTransport())
    url = client.url(
        "public/get_instruments", {"currency": "BTC", "kind": "option", "expired": False}
    )
    assert url.endswith("/public/get_instruments?currency=BTC&kind=option&expired=false")


@pytest.mark.parametrize(
    ("fixture", "code"),
    [("error_invalid_params.json", -32602), ("error_method_not_found.json", -32601)],
)
def test_client_errors_raise_without_retry(fixture, code):
    transport = ScriptedTransport(fixture_response(fixture, status=400))
    client, clock = make_client(transport)
    with pytest.raises(DeribitAPIError) as exc:
        client.call("public/ticker", instrument_name="BTC-NOPE")
    assert exc.value.code == code
    assert exc.value.http_status == 400
    assert len(transport.urls) == 1
    assert clock.sleeps == []


def test_invalid_params_error_keeps_deribit_detail():
    transport = ScriptedTransport(fixture_response("error_invalid_params.json", status=400))
    client, _ = make_client(transport)
    with pytest.raises(DeribitAPIError) as exc:
        client.call("public/ticker", instrument_name="BTC-NOPE")
    assert exc.value.data == {"reason": "wrong format", "param": "instrument_name"}


def test_server_errors_are_retried_with_growing_backoff():
    cloudflare = HttpResponse(502, b"<html>Bad gateway</html>")
    transport = ScriptedTransport(cloudflare, HttpResponse(503, b""), index_ok())
    client, clock = make_client(transport)
    result = client.call("public/get_index_price", index_name="btc_usd")
    assert result.result["index_price"] > 0
    assert len(transport.urls) == 3
    policy = RetryPolicy()
    assert len(clock.sleeps) == 2
    for retry, slept in enumerate(clock.sleeps):
        ceiling = min(policy.backoff_cap, policy.backoff_base * 2**retry)
        assert ceiling / 2 <= slept <= ceiling


@pytest.mark.parametrize("status", [400, 429])
def test_rate_limit_error_waits_at_least_the_minimum(status):
    transport = ScriptedTransport(rate_limit_error(status), index_ok())
    client, clock = make_client(transport)
    client.call("public/get_index_price", index_name="btc_usd")
    assert len(transport.urls) == 2
    assert clock.sleeps[0] >= RetryPolicy().rate_limited_min_wait


def test_http_429_respects_retry_after_header():
    limited = HttpResponse(429, b"slow down", {"retry-after": "7"})
    transport = ScriptedTransport(limited, index_ok())
    client, clock = make_client(transport)
    client.call("public/get_index_price", index_name="btc_usd")
    assert clock.sleeps[0] >= 7.0


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError(ConnectionRefusedError("refused")),
        TimeoutError("timed out"),
        ConnectionResetError("reset by peer"),
        http.client.IncompleteRead(b"partial"),
        http.client.RemoteDisconnected("closed"),
    ],
)
def test_network_errors_are_retried(failure):
    transport = ScriptedTransport(failure, index_ok())
    client, clock = make_client(transport)
    client.call("public/get_index_price", index_name="btc_usd")
    assert len(transport.urls) == 2
    assert len(clock.sleeps) == 1


def test_truncated_body_is_retried():
    transport = ScriptedTransport(HttpResponse(200, b'{"jsonrpc": "2.0", "res'), index_ok())
    client, _ = make_client(transport)
    client.call("public/get_index_price", index_name="btc_usd")
    assert len(transport.urls) == 2


def test_gives_up_after_max_retries():
    policy = RetryPolicy(max_retries=3)
    transport = ScriptedTransport(*[HttpResponse(503, b"")] * 4)
    client, clock = make_client(transport, retry=policy)
    with pytest.raises(DeribitRequestError, match=r"after 4 attempts.*HTTP 503"):
        client.call("public/get_index_price", index_name="btc_usd")
    assert len(transport.urls) == 4
    assert len(clock.sleeps) == 3


def test_certificate_failure_is_not_retried():
    bad_cert = urllib.error.URLError(ssl.SSLCertVerificationError("certificate verify failed"))
    transport = ScriptedTransport(bad_cert)
    client, _ = make_client(transport)
    with pytest.raises(DeribitRequestError, match="certificate"):
        client.call("public/get_index_price", index_name="btc_usd")
    assert len(transport.urls) == 1


def test_unexpected_http_status_without_json_raises():
    transport = ScriptedTransport(HttpResponse(403, b"<html>Forbidden</html>"))
    client, _ = make_client(transport)
    with pytest.raises(DeribitAPIError) as exc:
        client.call("public/get_index_price", index_name="btc_usd")
    assert exc.value.http_status == 403
    assert len(transport.urls) == 1


def test_backoff_never_exceeds_cap():
    policy = RetryPolicy(backoff_base=0.5, backoff_cap=4.0)
    rng = random.Random(1)
    delays = [policy.delay(retry, rng) for retry in range(12)]
    assert all(0.25 <= d <= 4.0 for d in delays)
    assert max(delays[6:]) <= 4.0


def test_token_bucket_allows_burst_then_throttles():
    clock = FakeClock()
    bucket = TokenBucket(RateLimit(rate=2.0, burst=3), clock.time, clock.sleep)
    assert [bucket.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert bucket.acquire() == pytest.approx(0.5)
    clock.now += 60.0  # a long pause refills the bucket, but only up to the burst size
    assert [bucket.acquire() for _ in range(3)] == [0.0, 0.0, 0.0]
    assert bucket.acquire() == pytest.approx(0.5)


def test_sustained_rate_matches_limit():
    clock = FakeClock()
    bucket = TokenBucket(RateLimit(rate=10.0, burst=20), clock.time, clock.sleep)
    start = clock.now
    for _ in range(220):
        bucket.acquire()
    # 20 requests ride on the initial burst, the other 200 come at 10 per second.
    assert clock.now - start == pytest.approx(20.0)


def test_method_specific_limit_applies_only_to_that_method():
    limits = {
        "default": RateLimit(rate=100.0, burst=100),
        "public/get_instruments": RateLimit(rate=1.0, burst=1),
    }
    instruments = fixture_response("get_instruments_btc_option.json")
    transport = ScriptedTransport(instruments, instruments, index_ok(), index_ok())
    client, clock = make_client(transport, rate_limits=limits)
    client.call("public/get_instruments", currency="BTC", kind="option", expired=False)
    client.call("public/get_instruments", currency="BTC", kind="option", expired=False)
    assert clock.sleeps == [pytest.approx(1.0)]
    client.call("public/get_index_price", index_name="btc_usd")
    client.call("public/get_index_price", index_name="btc_usd")
    assert clock.sleeps == [pytest.approx(1.0)]


def test_rate_limits_need_a_default_entry():
    with pytest.raises(ValueError, match="default"):
        DeribitClient(rate_limits={"public/ticker": RateLimit(1.0, 1)})


@pytest.mark.parametrize(("rate", "burst"), [(0.0, 1), (-1.0, 5), (1.0, 0)])
def test_invalid_rate_limit_is_rejected(rate, burst):
    with pytest.raises(ValueError, match="invalid rate limit"):
        RateLimit(rate=rate, burst=burst)
