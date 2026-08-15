"""Zero-dependency, tqdm-style progress bar for the training loops.

Avoids adding tqdm as a dependency: it prints a single carriage-return-updated
line to stderr showing an ASCII bar, the item count, percent, and an elapsed/ETA
estimate. When the total is unknown or output is not a TTY it degrades to a plain
periodic counter so redirected logs stay readable.
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

_BAR_WIDTH = 24


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def iter_progress(
    iterable: Iterable[T],
    *,
    prefix: str = "",
    total: int | None = None,
    stream=sys.stderr,
) -> Iterator[T]:
    """Wrap ``iterable`` yielding its items while rendering an in-place bar.

    ``prefix`` labels the bar (e.g. ``"seed0 theta=5 | task 1/5 epoch 2/10"``).
    ``total`` is the item count; if None it is taken from ``len(iterable)`` when
    available, else the bar shows only a running count.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except (TypeError, AttributeError):
            total = None

    is_tty = hasattr(stream, "isatty") and stream.isatty()
    start = time.monotonic()
    count = 0

    def render(done: bool) -> None:
        elapsed = time.monotonic() - start
        if total:
            frac = count / total
            filled = int(_BAR_WIDTH * frac)
            bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
            rate = count / elapsed if elapsed > 0 else 0.0
            eta = (total - count) / rate if rate > 0 else 0.0
            msg = (f"{prefix} [{bar}] {count}/{total} {frac*100:3.0f}% "
                   f"{_fmt_time(elapsed)}<{_fmt_time(eta)}")
        else:
            msg = f"{prefix} {count} items {_fmt_time(elapsed)}"
        if is_tty:
            stream.write("\r" + msg + "\x1b[K")
        else:
            stream.write(msg + "\n")
        stream.flush()

    for item in iterable:
        yield item
        count += 1
        # On a TTY refresh every item; otherwise only at the end to avoid log spam.
        if is_tty:
            render(done=False)

    render(done=True)
    if is_tty:
        stream.write("\n")
        stream.flush()
