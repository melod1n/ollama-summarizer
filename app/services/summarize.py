import asyncio
import json
import re
from time import time

import requests
from bs4 import BeautifulSoup
from fastapi import HTTPException
from readability import Document
from sqlalchemy.future import select

from app.core.config import MAX_TOKENS, CHUNK_MAX_TOKENS, CHUNK_OVERLAP
from app.core.logging import log
from app.db.models import Summary
from app.services.chunking import chunk_text
from app.services.ollama import call_ollama, build_prompt, filter_tags_via_llm
from main import encoding, task_status, queue_lock, task_queue


def process_queue_item(request_id: str):
    url = task_status[request_id]["request"]["url"]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_process_and_save(url, request_id))
    loop.close()


def fetch_and_clean_html(url: str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    html = response.text
    doc = Document(html)
    cleaned_html = doc.summary()
    soup = BeautifulSoup(cleaned_html, "html.parser")
    text = soup.get_text(strip=True)

    if not is_article(text):
        if not is_article_llm(text):
            raise HTTPException(status_code=400, detail="The provided URL does not contain a valid article.")

    return text


def is_article(text: str) -> bool:
    text = text.strip()

    if len(text) < 500:
        return False

    sentences = re.split(r'[.!?]', text)
    long_sentences = [s for s in sentences if len(s.strip().split()) > 6]
    if len(long_sentences) < 5:
        return False

    lower = text.lower()
    if any(term in lower for term in
           ["404", "page not found", "not found", "cookies", "consent", "login required", "sign in to continue"]):
        return False

    return True


def build_is_article_prompt(text: str) -> str:
    return f"""You are a web content classifier.

Determine whether the following page is a real article or not. An article should be at least one paragraph long, written in natural language, and contain meaningful content.

Only respond with a single word: "yes" or "no".

Here is the content:

{text[:2000]}"""


def is_article_llm(text: str) -> bool:
    prompt = build_is_article_prompt(text)
    log.info(f"[is_article_llm] Prompt:\n{prompt[:500]}...")

    try:
        response = call_ollama(prompt).strip().lower()
        log.info(f"[is_article_llm] LLM response: '{response}'")

        is_article = response.startswith("y")
        log.info(f"[is_article_llm] Final decision: {is_article}")
        return is_article

    except Exception as e:
        log.error(f"[is_article_llm] Error during LLM check: {e}")
        return False


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


async def _process_and_save(url: str, request_id: str):
    from app.db.database import async_session
    async with async_session() as session:
        start_time = time()
        log.info(f"⚙️ Start processing: request_id={request_id}, url={url}")

        try:
            text = fetch_and_clean_html(url)
            prompt = build_prompt(text)
            tokens = len(encoding.encode(prompt, disallowed_special=()))
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
