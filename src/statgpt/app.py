import logging
import os
import sys
from pathlib import Path

statgpt_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(statgpt_path))

import dotenv
from aidial_sdk import DIALApp

dotenv_path = os.path.join(os.getcwd(), ".env")

# noinspection PyBroadException
try:
    dotenv.load_dotenv(dotenv_path)
except Exception:
    pass

_log = logging.getLogger(__name__)

_log.info("Initializing StatGPT application")


def run_dial_app(app: DIALApp):
    import uvicorn

    uvicorn.run(app, port=5000, log_config=None)


from statgpt.application.app_factory import DialAppFactory

app_factory = DialAppFactory()
app = app_factory.create_app()


if __name__ == "__main__":
    from statgpt.settings.dial_app import AppMode, dial_app_settings

    if dial_app_settings.mode == AppMode.DIAL:
        pass
    elif dial_app_settings.mode == AppMode.LOCAL:
        run_dial_app(app)
    else:
        raise NotImplementedError(f"Unsupported application mode: {dial_app_settings.mode}")
