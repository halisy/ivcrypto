"""Minimal client for Deribit's public JSON RPC API (v2) over HTTPS.

Facts this module is built on, taken from the live API rather than assumed
(see docs/deribit_api.md for the full inventory):

* Requests are plain GETs, ``{base_url}/{method}?{params}``. A success is a JSON
  RPC envelope ``{"jsonrpc", "result", "usIn", "usOut", "usDiff", "testnet"}``
  where ``usIn`` and ``usOut`` are the server receive and send times in
  microseconds since the epoch.
* Errors come back with HTTP 400 and a body ``{"error": {"code", "message",
  "data"}}``. Deribit documents code 10028 (``too_many_requests``) for an
  exhausted rate limit. Responses carry no rate limit headers.
* Rate limits (https://docs.deribit.com/articles/rate-limits): non matching
  engine requests default to 20 requests per second with bursts of 100,
  ``public/get_instruments`` is limited to 1 request per second with bursts of
  50, and unauthenticated requests are limited per IP at an unpublished level.
  The defaults below stay at half the documented rates.
"""

from __future__ import annotations

import json
import logging
import random
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from http.client import HTTPException
from typing import Any

from ivcrypto import __version__

logger = logging.getLogger(__name__)

PRODUCTION_URL = "https://www.deribit.com/api/v2"
TESTNET_URL = "https://test.deribit.com/api/v2"
USER_AGENT = f"ivcrypto/{__version__} (+https://github.com/halisy/ivcrypto)"

TOO_MANY_REQUESTS = 10028
"""JSON RPC error code Deribit returns when the rate limit credits run out."""


class DeribitError(Exception):
    """Base class for errors raised by this client."""


class DeribitAPIError(DeribitError):
    """Deribit rejected the request with an error that retrying will not fix."""

    def __init__(
        self,
        method: str,
        code: int | None,
        message: str,
        data: Any = None,
        http_status: int | None = None,
    ) -> None:
        detail = f" ({data})" if data else ""
        super().__init__(f"{method}: error {code}: {message}{detail}")
        self.method = method
        self.code = code
        self.message = message
        self.data = data
        self.http_status = http_status


class DeribitRequestError(DeribitError):
    """The request could not be completed: network failure or retries exhausted."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


Transport = Callable[[str, float], HttpResponse]
"""Performs one GET of ``url`` with a timeout in seconds. Network failures raise
``OSError`` or ``http.client.HTTPException``; HTTP error statuses are returned."""


def urllib_transport(url: str, timeout: float) -> HttpResponse:
    """Default transport built on the standard library (honours HTTPS_PROXY)."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, response.read(), dict(response.headers.items()))
    except urllib.error.HTTPError as err:
        # Deribit reports JSON RPC errors as HTTP 400 with a JSON body: keep the body.
        try:
            body = err.read() if err.fp is not None else b""
        finally:
            err.close()
        headers = dict(err.headers.items()) if err.headers is not None else {}
        return HttpResponse(err.code, body, headers)


@dataclass(frozen=True)
class RpcResult:
    """The ``result`` of one successful call plus the server timestamps."""

    method: str
    params: Mapping[str, Any]
    result: Any
    us_in: int | None
    us_out: int | None
    testnet: bool


@dataclass(frozen=True)
class RateLimit:
    """Sustained ``rate`` in requests per second, with bursts of up to ``burst`` requests."""

    rate: float
    burst: int

    def __post_init__(self) -> None:
        if self.rate <= 0 or self.burst < 1:
            raise ValueError(f"invalid rate limit: rate={self.rate}, burst={self.burst}")


DEFAULT_RATE_LIMITS: Mapping[str, RateLimit] = {
    "default": RateLimit(rate=10.0, burst=20),
    "public/get_instruments": RateLimit(rate=0.5, burst=5),
}


class TokenBucket:
    """Holds at most ``burst`` tokens and refills at ``rate`` tokens per second.

    A request takes its token immediately; if the bucket was empty the balance
    goes negative and the caller sleeps until the refill has paid that debt
    back. Sleeping once for the exact debt avoids a wait and recheck loop,
    which can spin on floating point remainders smaller than the clock's
    resolution.
    """

    def __init__(
        self,
        limit: RateLimit,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.limit = limit
        self._clock = clock
        self._sleep = sleep
        self._tokens = float(limit.burst)
        self._updated = clock()

    def acquire(self) -> float:
        """Take one token, sleeping if none is available. Returns the seconds slept."""
        now = self._clock()
        refill = (now - self._updated) * self.limit.rate
        self._tokens = min(float(self.limit.burst), self._tokens + refill) - 1.0
        self._updated = now
        if self._tokens >= 0.0:
            return 0.0
        delay = -self._tokens / self.limit.rate
        self._sleep(delay)
        return delay


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with equal jitter.

    Retry ``n`` (0 based) waits ``d/2 + U(0, d/2)`` seconds with
    ``d = min(backoff_cap, backoff_base * 2**n)``, at least
    ``rate_limited_min_wait`` after a rate limit response, and at least the
    server's ``Retry-After`` when one is sent.
    """

    max_retries: int = 5
    backoff_base: float = 0.5
    backoff_cap: float = 30.0
    rate_limited_min_wait: float = 1.0

    def delay(
        self,
        retry: int,
        rng: random.Random,
        *,
        rate_limited: bool = False,
        retry_after: float | None = None,
    ) -> float:
        ceiling = min(self.backoff_cap, self.backoff_base * 2.0**retry)
        delay = ceiling / 2.0 + rng.uniform(0.0, ceiling / 2.0)
        if rate_limited:
            delay = max(delay, self.rate_limited_min_wait)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay


@dataclass(frozen=True)
class _Retryable:
    reason: str
    rate_limited: bool = False
    retry_after: float | None = None


class DeribitClient:
    """Rate limited, retrying client for Deribit's public API methods.

    ``transport``, ``clock``, ``sleep`` and ``seed`` exist so tests can drive the
    retry and throttling logic deterministically without a network.
    """

    def __init__(
        self,
        base_url: str = PRODUCTION_URL,
        *,
        timeout: float = 30.0,
        retry: RetryPolicy | None = None,
        rate_limits: Mapping[str, RateLimit] | None = None,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        seed: int | None = None,
    ) -> None:
        limits = dict(DEFAULT_RATE_LIMITS if rate_limits is None else rate_limits)
        if "default" not in limits:
            raise ValueError("rate_limits needs a 'default' entry")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry = retry if retry is not None else RetryPolicy()
        self._transport = transport if transport is not None else urllib_transport
        self._sleep = sleep
        self._rng = random.Random(seed)
        self._buckets = {name: TokenBucket(limit, clock, sleep) for name, limit in limits.items()}

    def url(self, method: str, params: Mapping[str, Any]) -> str:
        query = urllib.parse.urlencode({key: _encode(value) for key, value in params.items()})
        return f"{self.base_url}/{method}" + (f"?{query}" if query else "")

    def call(self, method: str, **params: Any) -> RpcResult:
        """Call ``method`` (for example ``public/get_index_price``) and return its result."""
        url = self.url(method, params)
        bucket = self._buckets.get(method, self._buckets["default"])
        problem = "no attempt made"
        for attempt in range(self.retry.max_retries + 1):
            bucket.acquire()
            outcome: RpcResult | _Retryable
            try:
                response = self._transport(url, self.timeout)
            except (OSError, HTTPException) as exc:
                if _is_certificate_error(exc):
                    raise DeribitRequestError(
                        f"{method}: TLS certificate verification failed: {exc}"
                    ) from exc
                outcome = _Retryable(f"network error: {exc!r}")
            else:
                outcome = self._interpret(method, params, response)
            if isinstance(outcome, RpcResult):
                return outcome
            problem = outcome.reason
            if attempt == self.retry.max_retries:
                break
            delay = self.retry.delay(
                attempt,
                self._rng,
                rate_limited=outcome.rate_limited,
                retry_after=outcome.retry_after,
            )
            logger.warning(
                "%s failed (%s); retry %d of %d in %.2f s",
                method,
                problem,
                attempt + 1,
                self.retry.max_retries,
                delay,
            )
            self._sleep(delay)
        raise DeribitRequestError(
            f"{method}: giving up after {self.retry.max_retries + 1} attempts; last problem: "
            f"{problem}"
        )

    @staticmethod
    def _interpret(
        method: str, params: Mapping[str, Any], response: HttpResponse
    ) -> RpcResult | _Retryable:
        payload = _parse_json(response.body)
        retry_after = _retry_after_seconds(response.headers)
        error = payload.get("error") if payload is not None else None
        if isinstance(error, dict):
            code = error.get("code")
            message = str(error.get("message", ""))
            if code == TOO_MANY_REQUESTS or response.status == 429:
                reason = f"rate limited (HTTP {response.status}, code {code})"
                return _Retryable(reason, rate_limited=True, retry_after=retry_after)
            if response.status >= 500:
                return _Retryable(f"HTTP {response.status}, error {code}: {message}")
            raise DeribitAPIError(method, code, message, error.get("data"), response.status)
        if response.status == 429:
            return _Retryable("rate limited (HTTP 429)", rate_limited=True, retry_after=retry_after)
        if response.status >= 500:
            return _Retryable(f"HTTP {response.status}")
        if response.status != 200:
            snippet = response.body[:200].decode("utf-8", errors="replace")
            raise DeribitAPIError(method, None, f"HTTP {response.status}", snippet, response.status)
        if payload is None or "result" not in payload:
            return _Retryable("malformed or truncated response body")
        return RpcResult(
            method=method,
            params=dict(params),
            result=payload["result"],
            us_in=_optional_int(payload.get("usIn")),
            us_out=_optional_int(payload.get("usOut")),
            testnet=bool(payload.get("testnet", False)),
        )


def _encode(value: Any) -> str:
    # Deribit expects lowercase booleans ("expired=false"); str(False) would send "False".
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_json(body: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    for name, value in headers.items():
        if name.lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except ValueError:  # an HTTP date; not worth parsing for our purposes
                return None
    return None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _is_certificate_error(exc: BaseException) -> bool:
    reason = getattr(exc, "reason", None)
    return isinstance(exc, ssl.SSLCertVerificationError) or isinstance(
        reason, ssl.SSLCertVerificationError
    )
