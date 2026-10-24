"""Fetches shared across analyses running side by side, on disk under ``data_cache_dir``.

Several tickers analysed at once each fetched the same global news, macro series
and prediction markets, and each hit Reddit's per-IP limit in the same minute
(nearly every run of a seven-ticker batch got HTTP 429). Two tools fix that:

- ``cached`` serves one fetch to every process that wants it within ``ttl``.
  The fetch runs under a per-key lock, so a second process asking for the same
  key while the first is fetching waits and then reads the result instead of
  fetching again. A result ``keep`` rejects (a failure, an "unavailable"
  placeholder) is returned but not stored, so a throttled moment isn't
  remembered for hours.
- ``paced`` spaces requests to one host at least ``interval`` seconds apart
  across processes, by recording the last request time under a lock.

Entries are JSON files named by a hash of the key; the cache needs no service.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.files import locked


def _root(*parts: str) -> str:
    path = os.path.join(get_config()["data_cache_dir"], "shared", *parts)
    os.makedirs(path, exist_ok=True)
    return path


def _key_file(namespace: str, key: Any) -> str:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return os.path.join(_root(namespace), f"{digest}.json")


def _read(path: str, ttl: float):
    try:
        with open(path, encoding="utf-8") as f:
            entry = json.load(f)
    except (OSError, ValueError):
        return None
    if time.time() - entry.get("fetched_at", 0) > ttl:
        return None
    return entry


def cached(namespace: str, key: Any, ttl: float, fetch: Callable[[], Any],
           keep: Callable[[Any], bool] = lambda value: value is not None) -> Any:
    """``fetch()``'s result for ``key``, from disk when another process fetched it within ``ttl``."""
    path = _key_file(namespace, key)
    entry = _read(path, ttl)
    if entry is not None:
        return entry["value"]
    with locked(path):
        entry = _read(path, ttl)  # someone may have fetched it while we waited
        if entry is not None:
            return entry["value"]
        value = fetch()
        if keep(value):
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"fetched_at": time.time(), "key": key, "value": value}, f, default=str)
            os.replace(tmp, path)
        return value


def prune(max_age_days: float = 7) -> int:
    """Delete entries nobody has refreshed in ``max_age_days``; returns how many went.

    Keys carry their dates, so an old entry is never asked for again; without
    this the cache only grows.
    """
    root = _root()
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for dirpath, _dirs, names in os.walk(root):
        if os.path.basename(dirpath) == "pace":
            continue
        for name in names:
            path = os.path.join(dirpath, name)
            if not name.endswith(".json"):
                continue
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
                    if os.path.exists(path + ".lock"):
                        os.remove(path + ".lock")
            except OSError:
                pass
    return removed


@contextmanager
def paced(host: str, interval: float):
    """Wait until ``interval`` seconds have passed since the last request to ``host`` by any process."""
    stamp = os.path.join(_root("pace"), host)
    with locked(stamp):
        try:
            with open(stamp, encoding="utf-8") as f:
                last = float(f.read().strip() or 0)
        except (OSError, ValueError):
            last = 0.0
        wait = last + interval - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            yield
        finally:
            with open(stamp, "w", encoding="utf-8") as f:
                f.write(str(time.time()))
