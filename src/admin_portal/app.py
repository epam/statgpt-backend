import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import dotenv
from fastapi import FastAPI

module_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(module_path))

dotenv_path = os.path.join(os.getcwd(), ".env")

# noinspection PyBroadException
try:
    dotenv.load_dotenv(dotenv_path)
except Exception:
    pass

from admin_portal.routers import router
from common.models import DatabaseHealthChecker, optional_msi_token_manager_context
from common.services.data_preloader import preload_data


@asynccontextmanager
async def lifespan(app_: FastAPI):
    async with optional_msi_token_manager_context():
        # Check resources' availability:
        await DatabaseHealthChecker.check()

        # Start data preloading in the background
        asyncio.create_task(preload_data())

        yield
        # Clean up


app = FastAPI(lifespan=lifespan)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, log_config=None)
