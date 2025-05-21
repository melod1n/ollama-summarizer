import asyncio
import json
import os
import threading
import logging
from typing import List, Dict
from uuid import uuid4
from time import time

import requests
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from readability import Document
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from chunking_utils import chunk_text
from database import engine, get_session
from models import Summary, Base

load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 6000))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", 5))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

app = FastAPI()
encoding = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

task_queue: List[str] = []
task_status: Dict[str, Dict] = {}
queue_lock = threading.Lock()


class URLRequest(BaseModel):
    url: str


class StatusResponse(BaseModel):
    status: str
    result: dict | None = None
    error: str | None = None


@app.on_event("startup")
async def on_startup():
    log.info("🟢 Backend started")
    log.info(
        f"Settings: OLLAMA_API_URL={OLLAMA_API_URL}, MODEL_NAME={MODEL_NAME}, MAX_TOKENS={MAX_TOKENS}, MAX_QUEUE_SIZE={MAX_QUEUE_SIZE}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


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


def process_queue_item(request_id: str):
    url = task_status[request_id]["request"]["url"]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_process_and_save(url, request_id))
    loop.close()


async def _process_and_save(url: str, request_id: str):
    from database import async_session
    async with async_session() as session:
        start_time = time()
        log.info(f"⚙️ Start processing: request_id={request_id}, url={url}")

        try:
            text = fetch_and_clean_html(url)
            prompt = build_prompt(text)
            tokens = len(encoding.encode(prompt))
            log.info(f"⚙️ Total symbols: {len(text)}, tokens: {len(encoding.encode(text))}, prompt tokens: {tokens}")

            if tokens > MAX_TOKENS:
                chunks = chunk_text(text, max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP)
                summaries = []
                all_tags_list = []
                for idx, chunk in enumerate(chunks):
                    log.info(
                        f"🧩 Chunk #{idx + 1} (symbols={len(chunk)}, tokens={len(encoding.encode(chunk))}):\n{chunk[:500]}...")
                    chunk_start_time = time()
                    chunk_prompt = build_prompt(chunk)
                    chunk_result = call_ollama(chunk_prompt)
                    log.info(
                        f"📨 LLM response for chunk #{idx + 1} (took {round(time() - chunk_start_time, 2)}s):\n{chunk_result}")
                    parsed = try_parse_result(chunk_result)
                    if "summary" in parsed:
                        summaries.append(parsed["summary"])
                    if "tags" in parsed:
                        all_tags_list.append(parsed.get("tags", []))

                final_tags = filter_tags_via_llm(all_tags_list)
                final_summary = summarize_chunk_summaries(summaries)
                final_result = {
                    "url": url,
                    "summary": final_summary,
                    "tags": final_tags,
                    "chunks": len(chunks)
                }
            else:
                result = call_ollama(prompt)
                log.info(f"📨 LLM response (single chunk):\n{result}")
                parsed = try_parse_result(result)
                final_result = {"url": url, **parsed}

            task_status[request_id]["status"] = "success"
            task_status[request_id]["result"] = final_result

            row = await session.execute(select(Summary).where(Summary.url == url))
            entry = row.scalar_one()
            entry.status = "success"
            entry.result = json.dumps(final_result)
            entry.duration_sec = round(time() - start_time, 2)
            entry.total_tokens = tokens
            await session.commit()
            log.info(f"✅ Success: request_id={request_id}, url={url}, duration={entry.duration_sec} sec")

        except Exception as e:
            task_status[request_id]["status"] = "failure"
            task_status[request_id]["error"] = str(e)
            row = await session.execute(select(Summary).where(Summary.url == url))
            entry = row.scalar_one()
            entry.status = "failure"
            entry.error = str(e)
            await session.commit()
            log.error(f"❌ Error processing request_id={request_id}, url={url}: {e}")

        finally:
            with queue_lock:
                if request_id in task_queue:
                    task_queue.remove(request_id)
            log.info(f"🧹 Finished processing: request_id={request_id}")


def fetch_and_clean_html(url: str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    html = response.text
    doc = Document(html)
    cleaned_html = doc.summary()
    soup = BeautifulSoup(cleaned_html, "html.parser")
    return soup.get_text(strip=True)


def build_prompt(text: str) -> str:
    return f"""You are a text analyzer.

    Given the following raw HTML of an article:
    1. Write a **concise topic-style summary** in **one English sentence** that reflects the article's subject.
       - Do **not** start with phrases like \"This article discusses...\" or \"The article explains...\".
       - Make it a **clear, compact statement** of the main idea.
       - Example: \"Best practices for handling display cutouts in Android edge-to-edge layouts.\"
    2. Generate **5 to 10 general-topic tags**, written in **English**, lowercase, and **hyphenated** (e.g. \"android\", \"mobile-development\", \"user-interface\").
       - Tags must describe the **overall subject area**, not specific technologies or methods.
       - Avoid concrete APIs or libraries (e.g. no \"recyclerview\", \"compose\").
       - Prefer broad tags like \"android\", \"mobile-ui\", \"design-principles\", \"user-experience\".

    Return the result only as valid JSON object, without wrapping it in markdown, code block, or any additional formatting. Just plain JSON. Like this:
    {{
      "summary": "...",
      "tags": ["...", "..."]
    }}

    Here is the HTML:
    {text}
    """


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()["response"].strip()


def try_parse_result(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "summary" in parsed and "tags" in parsed:
            return parsed
    except:
        pass
    try:
        cleaned = raw.strip('"').strip('```json').strip('```').strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "summary" in parsed and "tags" in parsed:
            return parsed
    except:
        pass
    return {"raw_response": raw, "error": "Failed to parse"}


def summarize_chunk_summaries(summaries: list[str]) -> str:
    if not summaries:
        return "No summaries available."
    if len(summaries) == 1:
        return summaries[0]
    unique_summaries = list(dict.fromkeys(summaries))
    joined = " ".join(unique_summaries)
    prompt = f"""Summarize the following partial summaries into one single coherent summary (in one English sentence):\n\n{joined}"""
    raw = call_ollama(prompt)
    final = try_parse_result(raw)
    summary = final.get("summary")
    if summary and len(summary) < 500:
        return summary
    return joined[:500].rsplit(".", 1)[0] + "..."


def filter_tags_via_llm(tag_lists: list[list[str]]) -> list[str]:
    flat_tags = [tag.strip().lower() for tags in tag_lists for tag in tags]
    unique_tags = sorted(set(flat_tags))
    prompt = f"""You are a text categorization assistant.

Here is a list of tags extracted from different parts of an article. They may contain duplicates, synonyms, or overly specific variations.

Your task is to:
- return **no more than 10** general-topic tags.
- remove tags that are too specific or repetitive in meaning.
- prefer broad, meaningful categories over concrete tools or libraries.
- keep the tags **in lowercase** and **hyphenated** (e.g. \"machine-learning\", \"language-models\").

Tags:
{json.dumps(unique_tags, indent=2)}

Return the result as a valid JSON array, like this:
["tag-one", "tag-two", "tag-three"]
"""
    raw = call_ollama(prompt).strip('"').strip('```json').strip('```').strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(tag, str) for tag in parsed):
            return parsed[:10]
    except Exception as e:
        log.warning(f"⚠️ Failed to parse tag response from LLM: {e}\nRaw: {raw}")
    return unique_tags[:10]
