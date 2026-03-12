import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from algotrade.api.scheduler import router as scheduler_router
from algotrade.api.scheduler import scheduler
from algotrade.settings import settings
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from structlog import get_logger

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    scheduler.start()
    logger.debug("Scheduler started")
    yield

    scheduler.shutdown()
    logger.debug("Scheduler shutdown")


app = FastAPI(lifespan=lifespan)
security = HTTPBasic(auto_error=False)


def get_current_username(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(security)],
):
    if not settings.basic_username or not settings.basic_password:
        return "anonymous"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    if not secrets.compare_digest(credentials.username, settings.basic_username) or not secrets.compare_digest(
        credentials.password, settings.basic_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


@app.get("/users/me")
def read_current_user(username: Annotated[str, Depends(get_current_username)]):
    return {"username": username}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    return {"status": "ok", "scheduler_running": bool(getattr(scheduler, "running", False))}


app.include_router(scheduler_router, dependencies=[Depends(get_current_username)])
