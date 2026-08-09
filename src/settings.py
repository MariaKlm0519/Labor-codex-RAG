from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
GROQ_MODEL    = "llama-3.1-8b-instant"

PROJECT_ROOT = Path(__file__).parent.parent

INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "tk_index.pkl"
DOCX_PATH = PROJECT_ROOT / "data" / "raw" / "Working_K.docx"
TEST_PATH = PROJECT_ROOT / "test" / "questions.json"

CHUNK_MAX_LEN = 1024