import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "ips.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    title         TEXT NOT NULL,
    text          TEXT NOT NULL,
    date          TEXT NOT NULL,          -- YYYY-MM-DD
    time          TEXT NOT NULL,          -- HH:MM:SS
    source        TEXT NOT NULL DEFAULT 'добавлен вручную'
);

CREATE TABLE IF NOT EXISTS lemmas (
    lemma_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    lemma     TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS index_terms (
    lemma_id     INTEGER NOT NULL REFERENCES lemmas(lemma_id) ON DELETE CASCADE,
    document_id  INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    frequency    INTEGER NOT NULL,       -- Q_ij: частота основы в документе
    PRIMARY KEY (lemma_id, document_id)
);

CREATE TABLE IF NOT EXISTS test_queries (
    query_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS relevance_judgments (
    query_id    INTEGER NOT NULL REFERENCES test_queries(query_id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    relevant    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (query_id, document_id)
);

CREATE INDEX IF NOT EXISTS idx_index_terms_doc ON index_terms(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_date ON documents(date);
"""


def get_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def total_documents():
    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
    finally:
        conn.close()


def total_lemmas():
    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) AS n FROM lemmas").fetchone()["n"]
    finally:
        conn.close()


def get_document_row(document_id):
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
    finally:
        conn.close()


def all_documents():
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM documents ORDER BY date DESC, document_id DESC"
        ).fetchall()
    finally:
        conn.close()


def add_test_query(query_text, description, relevant_ids):
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO test_queries(query_text, description) VALUES (?, ?)",
            (query_text, description),
        )
        qid = cur.lastrowid
        conn.executemany(
            "INSERT OR IGNORE INTO relevance_judgments(query_id, document_id, relevant)"
            " VALUES (?, ?, 1)",
            [(qid, d) for d in relevant_ids],
        )
        conn.commit()
        return qid
    finally:
        conn.close()


def delete_test_query(query_id):
    conn = get_connection()
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM test_queries WHERE query_id = ?", (query_id,))
        conn.commit()
    finally:
        conn.close()


def get_test_queries():
    conn = get_connection()
    try:
        return conn.execute(
            "SELECT * FROM test_queries ORDER BY query_id"
        ).fetchall()
    finally:
        conn.close()


def get_relevant_ids(query_id):
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT document_id FROM relevance_judgments"
            " WHERE query_id = ? AND relevant = 1",
            (query_id,),
        ).fetchall()
        return {r["document_id"] for r in rows}
    finally:
        conn.close()
