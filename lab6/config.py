import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATABASE_PATH = os.path.join(BASE_DIR, 'rag_database.db')
MAX_CONTENT_LENGTH = 64 * 1024 * 1024

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "granite4.1:8b"

DB_CONN_ARGS = {"check_same_thread": False}
