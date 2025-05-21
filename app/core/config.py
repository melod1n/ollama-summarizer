import os

from dotenv import load_dotenv

load_dotenv()

IN_DOCKER = os.getenv("IN_DOCKER", "false").lower() == "true"

if IN_DOCKER:
    DB_PATH = "/db/summaries.db"
    LOG_PATH = "/logs/summary.log"
else:
    DB_PATH = "./summaries.db"
    LOG_PATH = "./summary.log"

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 6000))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", 5))
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
