import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "syntax_lab.db"

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                order_num INTEGER NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                lemma TEXT,
                pos TEXT,
                tag TEXT,
                dep TEXT,
                head_text TEXT,
                head_pos TEXT,
                entity_type TEXT,
                entity_label TEXT,
                morph TEXT,
                wordnet TEXT,
                order_num INTEGER NOT NULL,
                FOREIGN KEY (sentence_id) REFERENCES sentences(id) ON DELETE CASCADE
            );
        """)
        conn.commit()

def save_analysis(filename, raw_text, sentences_data):
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO analyses (filename, raw_text) VALUES (?, ?)",
            (filename, raw_text)
        )
        analysis_id = cursor.lastrowid

        for s_idx, sent in enumerate(sentences_data):
            cursor = conn.execute(
                "INSERT INTO sentences (analysis_id, text, order_num) VALUES (?, ?, ?)",
                (analysis_id, sent["text"], s_idx)
            )
            sent_id = cursor.lastrowid

            for t_idx, tok in enumerate(sent["tokens"]):
                wn_json = json.dumps(tok.get("wordnet", []), ensure_ascii=False) if tok.get("wordnet") else None
                conn.execute("""
                    INSERT INTO tokens 
                    (sentence_id, text, lemma, pos, tag, dep, head_text, head_pos, 
                     entity_type, entity_label, morph, wordnet, order_num)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sent_id, tok["text"], tok.get("lemma"), tok.get("pos"), tok.get("tag"),
                    tok.get("dep"), tok.get("head_text"), tok.get("head_pos"),
                    tok.get("entity_type"), tok.get("entity_label"), tok.get("morph"),
                    wn_json, t_idx
                ))
        conn.commit()
        return analysis_id

def get_all_analyses():
    with get_db() as conn:
        return conn.execute(
            "SELECT id, filename, created_at, updated_at FROM analyses ORDER BY created_at DESC"
        ).fetchall()

def get_analysis(analysis_id):
    with get_db() as conn:
        analysis = conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
        if not analysis:
            return None

        sentences = conn.execute(
            "SELECT * FROM sentences WHERE analysis_id = ? ORDER BY order_num",
            (analysis_id,)
        ).fetchall()

        result = dict(analysis)
        result["sentences"] = []
        for sent in sentences:
            sent_dict = dict(sent)
            tokens = conn.execute(
                "SELECT * FROM tokens WHERE sentence_id = ? ORDER BY order_num",
                (sent["id"],)
            ).fetchall()
            sent_dict["tokens"] = []
            for t in tokens:
                t_dict = dict(t)
                if t_dict.get("wordnet"):
                    try:
                        t_dict["wordnet"] = json.loads(t_dict["wordnet"])
                    except:
                        t_dict["wordnet"] = []
                else:
                    t_dict["wordnet"] = []
                sent_dict["tokens"].append(t_dict)
            result["sentences"].append(sent_dict)
        return result

def update_analysis_text(analysis_id, new_text):
    with get_db() as conn:
        conn.execute(
            "UPDATE analyses SET raw_text = ?, updated_at = ? WHERE id = ?",
            (new_text, datetime.now().isoformat(), analysis_id)
        )
        conn.commit()

def delete_analysis(analysis_id):
    with get_db() as conn:
        conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        conn.commit()
