import logging
import time
from collections.abc import Callable


class OptionalTimer:
    start: float
    format: str
    printer: Callable[[str], None]
    enabled: bool

    def __init__(
        self,
        format: str = "Elapsed time: {time}",
        printer: Callable[[str], None] = print,
        enabled: bool = True,
    ):
        self.start = time.perf_counter()
        self.format = format
        self.printer = printer
        self.enabled = enabled

    def stop(self) -> float:
        if not self.enabled:
            return 0.0
        return time.perf_counter() - self.start

    def __str__(self) -> str:
        if not self.enabled:
            return "disabled"
        return f"{self.stop():.3f}s"

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        if self.enabled:
            self.printer(self.format.format(time=self))


_log = logging.getLogger(__name__)


def debug_timer(title: str) -> OptionalTimer:
    return OptionalTimer(
        format="timer." + title + ": {time}",
        printer=_log.debug,
        enabled=_log.isEnabledFor(logging.DEBUG),
    )
