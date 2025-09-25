import logging
import time
from collections.abc import Callable


class Timer:
    start: float
    format: str
    printer: Callable[[str], None]

    def __init__(
        self,
        format: str = "Elapsed time: {time}",
        printer: Callable[[str], None] = print,
    ):
        self.start = time.perf_counter()
        self.format = format
        self.printer = printer

    def stop(self) -> float:
        return time.perf_counter() - self.start

    def __str__(self) -> str:
        return f"{self.stop():.3f}s"

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self.printer(self.format.format(time=self))


_log = logging.getLogger(__name__)


def debug_timer(title: str) -> Timer:
    return Timer(format="timer." + title + ": {time}", printer=_log.debug)
