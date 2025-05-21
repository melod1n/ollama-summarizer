import threading
from typing import List, Dict

import tiktoken
from fastapi import FastAPI

from app.core.config import OLLAMA_API_URL, MODEL_NAME, MAX_TOKENS, MAX_QUEUE_SIZE, IN_DOCKER
from app.core.logging import log
from app.db.database import engine
from app.db.models import Base

app = FastAPI()
encoding = tiktoken.get_encoding("cl100k_base")

task_queue: List[str] = []
task_status: Dict[str, Dict] = {}
queue_lock = threading.Lock()

from app.api.summarize import router as summarize_router

app.include_router(summarize_router)


@app.on_event("startup")
async def on_startup():
    log.info("🟢 Backend started")
    log.info(
        f"Settings: IN_DOCKER={IN_DOCKER}\nOLLAMA_API_URL={OLLAMA_API_URL}\nMODEL_NAME={MODEL_NAME}\nMAX_TOKENS={MAX_TOKENS}\nMAX_QUEUE_SIZE={MAX_QUEUE_SIZE}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
