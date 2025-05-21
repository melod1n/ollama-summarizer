import threading
from uuid import uuid4

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import MAX_QUEUE_SIZE
from app.core.logging import log
from app.schemas.summary import StatusResponse, URLRequest
from app.services.summarize import process_queue_item
from app.db.database import get_session
from main import queue_lock, task_queue, task_status, app
from app.db.models import Summary


@app.post("/summarize")
async def queue_summary_task(request: URLRequest, session: AsyncSession = Depends(get_session)):
    log.info(f"📥 New request for URL: {request.url}")
    result = await session.execute(select(Summary).where(Summary.url == request.url))
    existing = result.scalar_one_or_none()

    if existing and existing.status == "success":
        log.info(f"⚠️ URL already successfully processed and result will be replaced: {request.url}")

    request_id = str(uuid4())

    with queue_lock:
        if len(task_queue) >= MAX_QUEUE_SIZE:
            log.warning(f"🚫 Queue full. Rejected URL: {request.url}")
            raise HTTPException(status_code=429, detail="Queue is full. Try again later.")

        task_queue.append(request_id)
        task_status[request_id] = {
            "status": "in_progress",
            "request": {"url": request.url}
        }

    if not existing:
        new_entry = Summary(url=request.url, status="in_progress")
        session.add(new_entry)
    else:
        existing.status = "in_progress"
        existing.result = None
        existing.error = None

    await session.commit()
    log.info(f"🟡 Added to queue: request_id={request_id}, url={request.url}")
    threading.Thread(target=process_queue_item, args=(request_id,), daemon=True).start()

    return {"request_id": request_id}


@app.get("/status/{request_id}", response_model=StatusResponse)
def get_status(request_id: str):
    if request_id not in task_status:
        raise HTTPException(status_code=404, detail="Request not found")

    entry = task_status[request_id]
    status = entry["status"]

    if status == "in_progress":
        return StatusResponse(status="in_progress")
    if status == "success":
        return StatusResponse(status="success", result=entry.get("result"))
    return StatusResponse(status="failure", error=entry.get("error"))
