import json

import requests

from app.core.config import MODEL_NAME, OLLAMA_API_URL
from app.core.logging import log


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()["response"].strip()


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
