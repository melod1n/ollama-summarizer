from main import encoding


def chunk_text(text: str, max_tokens: int, overlap: int):
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokens[start:end]
        chunks.append(encoding.decode(chunk))
        start += max_tokens - overlap

    return chunks
