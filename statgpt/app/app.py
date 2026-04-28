import logging
import os
import sys
from pathlib import Path

statgpt_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(statgpt_path))

import dotenv

dotenv_path = os.path.join(os.getcwd(), ".env")

# noinspection PyBroadException
try:
    dotenv.load_dotenv(dotenv_path)
except Exception:
    pass

_log = logging.getLogger(__name__)

_log.info("Initializing StatGPT application")

from aidial_sdk import DIALApp


def run_dial_app(app: DIALApp):
    import uvicorn

    uvicorn.run(app, port=5000, log_config=None)


os.environ["OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT"] = "16384"

from statgpt.app.application.app_factory import DialAppFactory

app_factory = DialAppFactory()
app = app_factory.create_app()


if __name__ == "__main__":
    run_dial_app(app)
