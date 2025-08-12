import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from io import TextIOBase
from time import perf_counter

from aidial_sdk.chat_completion import Choice, Stage

from statgpt.config import DialAppConfig


class ContentStream(TextIOBase):
    def __init__(self, destination: Choice | Stage):
        self.destination = destination

    def write(self, content):
        self.destination.append_content(content)


async def periodic_ping(file: ContentStream, interval: int = 15):
    while True:
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
        print("", file=file, flush=True)


@contextmanager
def continuous_choice(choice: Choice, interval: int = 15):
    choice_io = ContentStream(choice)
    ping_task = asyncio.create_task(periodic_ping(choice_io, interval))
    try:
        yield choice
    finally:
        ping_task.cancel()


@contextmanager
def timed_stage(choice: Choice, *args, **kwargs):
    with choice.create_stage(*args, **kwargs) as stage:
        start = perf_counter()
        try:
            yield stage
        finally:
            end = perf_counter()

            if DialAppConfig.SHOW_STAGE_SECONDS:
                stage.append_name(f" ({end - start:.2f}s)")


@contextmanager
def optional_stage(stage_generator: Generator, enabled: bool):
    if not enabled:
        # Create a dummy stage that does nothing
        queue = asyncio.Queue()
        stage_generator = Stage(queue, 0, 0, name="A dummy stage")

    with stage_generator as stage:
        yield stage
