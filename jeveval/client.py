"""The API client: adaptive concurrency, retry with backoff, and raw logging.

Three requirements from the plan shape this module.

*Everything is logged before anything is computed.* Every attempt -- including
each retry and each failure -- is appended to a JSONL file as it happens, with
the full request and response bodies. All analysis runs offline against that
log, so a metric can be redefined and recomputed without re-spending calls.

*A failed call never becomes a data point.* A call that exhausts its retries
returns a `CallResult` with `ok=False` and no answers. Experiments count those
and exclude them; nothing is defaulted to 0.5, to False, or to the majority
class, because a defaulted failure is indistinguishable from a real answer once
it reaches a metric.

*Rate limits bind before cost does.* At $42 per billion input tokens the whole
plan costs a few dollars, but early-access rate limits are undocumented. The
limiter therefore starts conservative, increases additively while calls succeed,
and halves on the first 429 or 5xx -- and the sustained rate it achieves is
itself a reported finding.

Transport is `http.client` from the standard library, with one keep-alive
connection per worker thread, rather than httpx. This machine runs Guix, whose
Python profile precedes the virtualenv on `sys.path`, so a pip-installed package
can be shadowed by an older Guix copy of one of its dependencies -- which is
exactly what happened with httpx by way of anyio and typing_extensions. The
request this harness makes is one JSON POST, so depending on nothing but the
standard library removes a whole class of environment failure for no real cost.
"""

from __future__ import annotations

import http.client
import json
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

from . import config, wire
from .auth import get_api_key, redact


# --------------------------------------------------------------------------
# Call specification and result
# --------------------------------------------------------------------------


@dataclass
class Call:
    """One request to make.

    `experiment` and `condition` are what the offline analysis groups by, so
    they are required rather than optional metadata -- a logged call that cannot
    be attributed to a condition is not recoverable later.
    """

    experiment: str
    condition: str
    state: Any
    questions: dict[str, dict]
    instance_id: str = ""
    repetition: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class CallResult:
    call: Call
    ok: bool
    answers: dict[str, dict] = field(default_factory=dict)
    model_version: str | None = None
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    http_status: int | None = None
    error: str | None = None
    attempts: int = 1
    request_id: str | None = None
    raw_response: dict | None = None

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * config.USD_PER_INPUT_TOKEN
            + self.output_tokens * config.USD_PER_OUTPUT_TOKEN
        )


# --------------------------------------------------------------------------
# Adaptive concurrency
# --------------------------------------------------------------------------


class AdaptiveLimiter:
    """Additive-increase / multiplicative-decrease gate on in-flight requests.

    The worker pool is sized to the maximum concurrency once; this gate is what
    actually decides how many of those workers may have a request outstanding at
    any moment, so the limit can move up and down during a run.
    """

    def __init__(self, rate: config.RateConfig) -> None:
        self._rate = rate
        self.limit = rate.start_concurrency
        self._in_flight = 0
        self._successes = 0
        self._cond = threading.Condition()
        self.peak_limit = self.limit
        self.throttle_events = 0
        self.throttle_episodes = 0
        self._last_decrease = 0.0

    def acquire(self) -> int:
        with self._cond:
            while self._in_flight >= self.limit:
                self._cond.wait(timeout=1.0)
            self._in_flight += 1
            return self.limit

    def release(self) -> None:
        with self._cond:
            self._in_flight -= 1
            self._cond.notify()

    def on_success(self) -> None:
        with self._cond:
            self._successes += 1
            # Climb a step once a number of successes proportional to the
            # current limit have landed, so recovery takes roughly a constant
            # number of round trips rather than a constant number of calls. With
            # a flat threshold, a limit knocked down to 1 needed 50 successes at
            # about 2 calls/s to regain a single step -- twenty minutes to climb
            # back to where it started, which is slower than the throttling it
            # was reacting to.
            need = max(self._rate.min_successes_to_increase, self.limit)
            if self._successes >= need and self.limit < self._rate.max_concurrency:
                self._successes = 0
                self.limit += max(1, self.limit // 8)
                self.limit = min(self.limit, self._rate.max_concurrency)
                self.peak_limit = max(self.peak_limit, self.limit)
                self._cond.notify_all()

    def on_throttle(self) -> None:
        with self._cond:
            self.throttle_events += 1
            now = time.time()
            # Every request already in flight when the endpoint starts throttling
            # tends to come back throttled, so a burst of N rejections is one
            # event observed N times. Halving per rejection compounds it: four
            # 429s out of 533 calls took the limit from 128 to 1. Collapse a
            # burst into a single decrease by ignoring further throttles until
            # the in-flight requests from before the cut have had time to drain.
            if now - self._last_decrease < self._rate.throttle_cooldown:
                return
            self._last_decrease = now
            self.throttle_episodes += 1
            self._successes = 0
            self.limit = max(
                self._rate.min_concurrency, int(self.limit * self._rate.backoff_factor)
            )


# --------------------------------------------------------------------------
# Raw log
# --------------------------------------------------------------------------


class RunLog:
    """Append-only JSONL of every attempt.

    Written and flushed synchronously per record. The cost of a flush is
    irrelevant next to a 170ms network call, and it means a run killed halfway
    leaves a complete, parseable log of everything it did.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.records = 0

    def write(self, record: dict) -> None:
        line = redact(json.dumps(record, default=str))
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            self.records += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}

# Statuses that will not come right by retrying or by waiting, and that every
# subsequent call in the run will hit identically: an exhausted credit balance,
# a rejected key, a revoked one. Continuing past the first of these produces
# nothing but a log full of identical failures -- an out-of-credits balance
# discovered partway through E3 burned 3,488 calls before anything noticed.
FATAL_STATUSES = {401, 402, 403}


def _rubric_ids(questions: dict[str, dict]) -> dict[str, list[str]]:
    """Rubric ids per score question, in wire order, or {} if there are none."""
    out: dict[str, list[str]] = {}
    for key, q in questions.items():
        if q.get("type") == "score":
            out[key] = [str(r["id"]) for r in (q.get("rubric") or [])]
    return out


class JevClient:
    def __init__(
        self,
        *,
        run_dir: Path,
        model: str = config.MODEL_ALIAS,
        rate: config.RateConfig | None = None,
        log_name: str = "calls.jsonl",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.model = model
        self.rate = rate or config.RATE
        self.log = RunLog(self.run_dir / log_name)
        self.limiter = AdaptiveLimiter(self.rate)

        parts = urlsplit(config.API_URL)
        self._host = parts.netloc
        self._path = parts.path or "/"
        # The scheme is honoured rather than assumed. The hosted endpoint is
        # https, but an implementation of the same protocol running on this
        # machine is plain http, and refusing to speak to it would make the
        # harness the reason a comparison could not be run.
        self._https = parts.scheme != "http"
        self._ssl = ssl.create_default_context() if self._https else None
        self._local = threading.local()
        self._key: str | None = None
        self._fatal: str | None = None
        self._fatal_status: int | None = None

        # Run-level counters the report header needs.
        self._counter_lock = threading.Lock()
        self.total_calls = 0
        self.failed_calls = 0
        self.total_retries = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.model_versions: set[str] = set()
        self.stale_reconnects = 0
        self.status_counts: dict[str, int] = {}
        self._started: float | None = None

    def __enter__(self) -> "JevClient":
        self._key = get_api_key()
        self._started = time.time()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._close_connections()
        self.log.close()

    # -- connection management -------------------------------------------

    # Errors that mean "the peer closed an idle keep-alive socket", as opposed
    # to "the peer is overloaded". The two are indistinguishable at the status
    # level -- both arrive as no status at all -- and conflating them is how a
    # healthy server ends up being treated as a throttling one.
    STALE_CONNECTION = (BrokenPipeError, ConnectionResetError,
                        http.client.RemoteDisconnected, http.client.BadStatusLine)

    def _conn(self) -> tuple[http.client.HTTPConnection, bool]:
        """One keep-alive connection per worker thread, and whether it is new."""
        conn = getattr(self._local, "conn", None)
        fresh = conn is None
        if conn is None:
            if self._https:
                conn = http.client.HTTPSConnection(
                    self._host, context=self._ssl, timeout=self.rate.request_timeout
                )
            else:
                conn = http.client.HTTPConnection(
                    self._host, timeout=self.rate.request_timeout
                )
            self._local.conn = conn
        return conn, fresh

    def _drop_connection(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _close_connections(self) -> None:
        self._drop_connection()

    def _post(self, body: bytes) -> tuple[int, bytes, dict]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._key}",
            "Accept": "application/json",
            "Content-Length": str(len(body)),
        }
        # A server is free to close an idle keep-alive connection whenever it
        # likes, and the client only finds out by writing to it. That is not an
        # error worth reporting: reconnect and send again. Only a reused
        # connection gets this second chance, so a genuinely unreachable server
        # still fails on its first attempt instead of being retried twice.
        for attempt in (1, 2):
            conn, fresh = self._conn()
            try:
                conn.request("POST", self._path, body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                return resp.status, data, dict(resp.getheaders())
            except Exception as exc:
                # A dead connection must not poison every later call on this
                # thread, so drop it and let the next attempt build a fresh one.
                self._drop_connection()
                if attempt == 1 and not fresh and isinstance(exc, self.STALE_CONNECTION):
                    with self._counter_lock:
                        self.stale_reconnects += 1
                    continue
                raise
        raise AssertionError("unreachable")

    # -- one call ---------------------------------------------------------

    def call(self, spec: Call) -> CallResult:
        # Once the run has hit an unrecoverable condition, every later call
        # would fail the same way. Return without touching the network, and
        # count it as a failure so the totals stay honest about what was not
        # measured.
        if self._fatal is not None:
            self._tally(failed=True)
            return CallResult(
                call=spec, ok=False, http_status=self._fatal_status,
                error=f"aborted before sending: {self._fatal}",
            )

        body_obj = wire.build_request(
            state=spec.state, questions=spec.questions, model=self.model
        )
        body = json.dumps(body_obj).encode("utf-8")

        delay = self.rate.base_retry_delay
        last_error: str | None = None
        last_status: int | None = None

        for attempt in range(1, self.rate.max_retries + 1):
            concurrency = self.limiter.acquire()
            t0 = time.time()
            status: int | None = None
            payload: Any = None
            headers: dict = {}
            transport_error: str | None = None
            try:
                status, raw, headers = self._post(body)
                try:
                    payload = json.loads(raw)
                except Exception:
                    payload = {"_unparseable_body": raw.decode("utf-8", "replace")[:2000]}
            except Exception as exc:
                transport_error = f"{type(exc).__name__}: {exc}"
            finally:
                latency = time.time() - t0
                self.limiter.release()

            request_id = headers.get("x-typesafe-request-id")
            record = {
                "ts": t0,
                "experiment": spec.experiment,
                "condition": spec.condition,
                "instance_id": spec.instance_id,
                "repetition": spec.repetition,
                # The wire form loses one thing the caller needs back: a score
                # question's rubric travels as a list of labels, so the caller's
                # rubric ids -- which are what ground truth is expressed in --
                # are not recoverable from the request alone. Without this the
                # log cannot rescore a score question offline, which the plan
                # requires. Choice questions need no equivalent, because their
                # option ids are the criteria keys.
                "rubric_ids": _rubric_ids(spec.questions),
                "attempt": attempt,
                "http_status": status,
                "request_id": request_id,
                "latency_s": latency,
                "concurrency": concurrency,
                "request": body_obj,
                "response": payload,
                "transport_error": transport_error,
                "meta": spec.meta,
            }
            self._count_status(transport_error or str(status))

            # Transport failure or retryable status.
            if transport_error is not None or status in RETRY_STATUSES:
                last_error = transport_error or f"HTTP {status}"
                last_status = status
                record["outcome"] = "retry"
                self.log.write(record)
                # Back off for backpressure, not for a socket the peer closed.
                # A timeout does mean the server is struggling, so it counts.
                overloaded = status in (429, 503) or (
                    status is None and transport_error is not None
                    and not transport_error.startswith(
                        tuple(e.__name__ for e in self.STALE_CONNECTION))
                )
                if overloaded:
                    self.limiter.on_throttle()
                if attempt >= self.rate.max_retries:
                    break
                with self._counter_lock:
                    self.total_retries += 1
                wait = delay
                ra = headers.get("retry-after")
                if ra:
                    try:
                        wait = max(wait, float(ra))
                    except ValueError:
                        pass
                time.sleep(min(wait, self.rate.max_retry_delay))
                delay = min(delay * 2, self.rate.max_retry_delay)
                continue

            # Non-retryable error.
            if status != 200:
                if status in FATAL_STATUSES and self._fatal is None:
                    detail = ""
                    if isinstance(payload, dict):
                        d = payload.get("detail")
                        detail = (d or {}).get("message", "") if isinstance(d, dict) else str(d or "")
                    self._fatal = f"HTTP {status}: {detail or 'unrecoverable'}"
                    self._fatal_status = status
                    print(
                        f"\n  ABORTING RUN -- {self._fatal}\n"
                        f"  Every remaining call would fail identically. "
                        f"{self.total_calls} calls completed so far.\n",
                        flush=True,
                    )
                record["outcome"] = "error"
                self.log.write(record)
                self._tally(failed=True)
                return CallResult(
                    call=spec,
                    ok=False,
                    http_status=status,
                    error=json.dumps(payload)[:500] if payload else f"HTTP {status}",
                    latency_s=latency,
                    attempts=attempt,
                    request_id=request_id,
                    raw_response=payload if isinstance(payload, dict) else None,
                )

            # Success -- but a response we cannot parse is a failure, not a
            # data point.
            try:
                answers = wire.parse_response(payload, spec.questions)
            except Exception as exc:
                record["outcome"] = "parse_error"
                record["parse_error"] = str(exc)
                self.log.write(record)
                self._tally(failed=True)
                return CallResult(
                    call=spec,
                    ok=False,
                    http_status=status,
                    error=f"parse: {exc}",
                    latency_s=latency,
                    attempts=attempt,
                    request_id=request_id,
                    raw_response=payload,
                )

            itok, otok = wire.usage(payload)
            version = wire.model_version(payload)
            record["outcome"] = "ok"
            record["input_tokens"] = itok
            record["output_tokens"] = otok
            record["model_version"] = version
            self.log.write(record)

            self.limiter.on_success()
            self._tally(failed=False, itok=itok, otok=otok, version=version)

            return CallResult(
                call=spec,
                ok=True,
                answers=answers,
                model_version=version,
                latency_s=latency,
                input_tokens=itok,
                output_tokens=otok,
                http_status=status,
                attempts=attempt,
                request_id=request_id,
                raw_response=payload,
            )

        # Retries exhausted. This needs its own terminal record: the attempts
        # above were all logged with outcome "retry", which the offline analysis
        # counts as retry history rather than as a call. Without this line the
        # log shows three retries and zero failures for a call the client
        # counted as failed -- and a run that loses calls to sustained rate
        # limiting would report "0 failed", which is the failure mode the plan
        # says is most likely to actually occur.
        self.log.write(
            {
                "ts": time.time(),
                "experiment": spec.experiment,
                "condition": spec.condition,
                "instance_id": spec.instance_id,
                "repetition": spec.repetition,
                "rubric_ids": _rubric_ids(spec.questions),
                "attempt": self.rate.max_retries,
                "http_status": last_status,
                "request_id": None,
                "latency_s": 0.0,
                "concurrency": self.limiter.limit,
                "request": body_obj,
                "response": None,
                "transport_error": last_error,
                "meta": spec.meta,
                "outcome": "error",
                "error": f"retries exhausted: {last_error}",
            }
        )
        self._tally(failed=True)
        return CallResult(
            call=spec,
            ok=False,
            http_status=last_status,
            error=f"retries exhausted: {last_error}",
            attempts=self.rate.max_retries,
        )

    def _tally(
        self,
        *,
        failed: bool,
        itok: int = 0,
        otok: int = 0,
        version: str | None = None,
    ) -> None:
        with self._counter_lock:
            self.total_calls += 1
            if failed:
                self.failed_calls += 1
            self.input_tokens += itok
            self.output_tokens += otok
            if version:
                self.model_versions.add(version)

    def _count_status(self, key: str) -> None:
        with self._counter_lock:
            self.status_counts[key] = self.status_counts.get(key, 0) + 1

    # -- many calls -------------------------------------------------------

    def run(
        self,
        specs: Sequence[Call] | Iterable[Call],
        *,
        progress_every: int = 250,
        label: str = "",
    ) -> list[CallResult]:
        """Run many calls concurrently, preserving input order in the output."""
        specs = list(specs)
        if not specs:
            return []
        results: list[CallResult | None] = [None] * len(specs)
        t0 = time.time()
        done_lock = threading.Lock()
        done = 0

        def one(i: int) -> None:
            nonlocal done
            results[i] = self.call(specs[i])
            with done_lock:
                done += 1
                d = done
            if progress_every and d % progress_every == 0:
                rate = d / max(1e-9, time.time() - t0)
                print(
                    f"  [{label or specs[0].experiment}] {d}/{len(specs)} "
                    f"{rate:.1f} req/s  concurrency={self.limiter.limit}  "
                    f"failed={self.failed_calls}",
                    flush=True,
                )

        workers = max(1, min(self.rate.max_concurrency, len(specs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, range(len(specs))))
        return [r for r in results if r is not None]

    # -- run-level summary -----------------------------------------------

    def summary(self) -> dict:
        elapsed = time.time() - (self._started or time.time())
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "total_retries": self.total_retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.input_tokens * config.USD_PER_INPUT_TOKEN,
            "model_versions": sorted(self.model_versions),
            "elapsed_s": elapsed,
            "sustained_req_per_s": self.total_calls / elapsed if elapsed > 0 else 0.0,
            "peak_concurrency": self.limiter.peak_limit,
            "final_concurrency": self.limiter.limit,
            "throttle_events": self.limiter.throttle_events,
            "throttle_episodes": self.limiter.throttle_episodes,
            "aborted": self._fatal,
            "status_counts": dict(self.status_counts),
            "log_path": str(self.log.path),
            "log_records": self.log.records,
        }
