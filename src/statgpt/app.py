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


from common.config import logger

logger.info("Initializing Talk-To-Your-Data application")


def run_dial_app(app: DIALApp):
    import uvicorn

    uvicorn.run(app, port=5000, log_config=None)


from statgpt.application.app_factory import DialAppFactory

app_factory = DialAppFactory()
app = app_factory.create_app()


if __name__ == "__main__":
    from statgpt.settings.application import AppMode, application_settings

    if application_settings.mode == AppMode.DIAL:
        pass
    elif application_settings.mode == AppMode.LOCAL:
        run_dial_app(app)
    else:
        raise NotImplementedError(f"Unsupported application mode: {application_settings.mode}")
