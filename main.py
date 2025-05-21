from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from readability import Document
from bs4 import BeautifulSoup
from fastapi.responses import JSONResponse
import json
import tiktoken
from typing import List
from chunking_utils import chunk_text
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "7500"))

app = FastAPI()
encoding = tiktoken.get_encoding("cl100k_base")

class URLRequest(BaseModel):
    urls: List[str]

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
    joined = " ".join(summaries)
    prompt = f"""Summarize the following partial summaries into one single coherent summary (in one English sentence):

{joined}
"""
    raw = call_ollama(prompt)
    final = try_parse_result(raw)
    return final.get("summary", joined[:500] + "...")


def fetch_and_clean_html(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    doc = Document(html)
    cleaned_html = doc.summary()
    soup = BeautifulSoup(cleaned_html, "html.parser")
    return soup.get_text(strip=True)

def build_prompt(text: str) -> str:
    return f"""You are a text analyzer.

Given the following raw HTML of an article:
1. Write a **concise topic-style summary** in **one English sentence** that reflects the article's subject.
   - Do **not** start with phrases like "This article discusses..." or "The article explains...".
   - Make it a **clear, compact statement** of the main idea.
   - Example: "Best practices for handling display cutouts in Android edge-to-edge layouts."
2. Generate **5 to 10 general-topic tags**, written in **English**, lowercase, and **hyphenated** (e.g. `"android"`, `"mobile-development"`, `"user-interface"`).
   - Tags must describe the **overall subject area**, not specific technologies or methods.
   - Avoid concrete APIs or libraries (e.g. no `"recyclerview"`, `"compose"`).
   - Prefer broad tags like `"android"`, `"mobile-ui"`, `"design-principles"`, `"user-experience"`.

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
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama request failed: {e}")

@app.post("/summarize")
def summarize(request: URLRequest):
    results = []
    for url in request.urls:
        try:
            text = fetch_and_clean_html(url)
            prompt = build_prompt(text)
            tokens = len(encoding.encode(prompt))

            if tokens > MAX_TOKENS:
                # chunking fallback
                chunks = chunk_text(text, max_tokens=1500, overlap=200)
                summaries = []
                all_tags = set()
                for idx, chunk in enumerate(chunks):
                    chunk_prompt = build_prompt(chunk)
                    chunk_result = call_ollama(chunk_prompt)
                    parsed = try_parse_result(chunk_result)
                    if "summary" in parsed:
                        summaries.append(parsed["summary"])
                    if "tags" in parsed:
                        all_tags.update(parsed.get("tags", []))
                final_summary = summarize_chunk_summaries(summaries)
                results.append({
                    "url": url,
                    "summary": final_summary,
                    "tags": sorted(all_tags),
                    "chunks": len(chunks)
                })
                continue

            result = call_ollama(prompt)

            # Попытка распарсить JSON напрямую
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict) and "summary" in parsed and "tags" in parsed:
                    results.append({"url": url, **parsed})
                    continue
            except json.JSONDecodeError:
                pass

            # Вторая попытка — иногда JSON внутри строки
            try:
                cleaned = result.strip('"').strip('```json').strip('`').strip('\n')
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "summary" in parsed and "tags" in parsed:
                    results.append({"url": url, **parsed})
                    continue
            except Exception:
                pass

            results.append({"url": url, "raw_response": result, "error": "Failed to parse valid JSON from LLM response"})

        except HTTPException as e:
            results.append({"url": url, "error": e.detail})
        except Exception as e:
            results.append({"url": url, "error": str(e)})

    return results
