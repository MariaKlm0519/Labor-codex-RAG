from dotenv import load_dotenv
import os

load_dotenv()

BOT_TOKEN = os.environ["BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
GROQ_MODEL    = "llama-3.1-8b-instant"

INDEX_PATH = "../data/processed/tk_index.pkl"
DOCX_PATH = "../data/raw/Working_K.docx"
TEST_PATH = "../test/questions.json"

CHUNK_MAX_LEN = 1024