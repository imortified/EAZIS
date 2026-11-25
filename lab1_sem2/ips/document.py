import datetime
import math
import re
from collections import Counter

from . import db
from .stemmer import stem_word

TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

STOPWORDS = {
    "и", "или", "не", "в", "во", "на", "с", "со", "от", "по", "для", "то",
    "что", "это", "эта", "этот", "is", "of", "as", "при", "над", "под",
    "между", "также", "их", "его", "её", "ее", "когда", "чем", "так", "уже",
    "еще", "ещё", "все", "весь", "быть", "был", "была", "было", "будут",
    "как", "но", "а", "да", "же", "ли", "либо", "если", "то", "там", "тут",
    "без", "до", "из", "за", "к", "о", "об", "у", "по", "про", "через",
    "the", "a", "an", "and", "or", "not",
}


def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def analyze(text):
    counts = Counter()
    for token in tokenize(text):
        if len(token) < 2 or token in STOPWORDS:
            continue
        counts[stem_word(token)] += 1
    return counts


class Document:

    def __init__(self, title, text, document_id=None, date=None, time=None,
                 source="добавлен вручную"):
        self.title = (title or "Без названия").strip()
        self.text = text or ""
        self.document_id = document_id
        now = datetime.datetime.now()
        self.date = date or now.strftime("%Y-%m-%d")
        self.time = time or now.strftime("%H:%M:%S")
        self.source = source

    def AddDocumentToBase(self):
        if not self.title or not self.text.strip():
            return None
        freq = analyze(self.title + "\n" + self.text)
        if not freq:
            return None

        conn = db.get_connection()
        try:
            cur = conn.execute(
                "INSERT INTO documents(title, text, date, time, source)"
                " VALUES (?, ?, ?, ?, ?)",
                (self.title, self.text, self.date, self.time, self.source),
            )
            self.document_id = cur.lastrowid
            for lemma, q in freq.items():
                row = conn.execute(
                    "SELECT lemma_id FROM lemmas WHERE lemma = ?", (lemma,)
                ).fetchone()
                if row is not None:
                    lemma_id = row["lemma_id"]
                else:
                    lemma_id = conn.execute(
                        "INSERT INTO lemmas(lemma) VALUES (?)", (lemma,)
                    ).lastrowid
                conn.execute(
                    "INSERT OR REPLACE INTO index_terms(lemma_id, document_id, frequency)"
                    " VALUES (?, ?, ?)",
                    (lemma_id, self.document_id, q),
                )
            conn.commit()
            return self.document_id
        finally:
            conn.close()

    @staticmethod
    def DeleteDocumentFromBase(document_id):
        conn = db.get_connection()
        try:
            cur = conn.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    @staticmethod
    def GetLemmInverseFrequency(lemma):
        conn = db.get_connection()
        try:
            if isinstance(lemma, int):
                row = conn.execute(
                    "SELECT l.lemma AS lemma FROM lemmas l WHERE l.lemma_id = ?",
                    (lemma,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT l.lemma AS lemma FROM lemmas l WHERE l.lemma = ?",
                    (lemma,),
                ).fetchone()
            if row is None:
                return None
            n = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            p = conn.execute(
                "SELECT COUNT(DISTINCT document_id) AS p FROM index_terms it"
                " JOIN lemmas l ON l.lemma_id = it.lemma_id WHERE l.lemma = ?",
                (row["lemma"],),
            ).fetchone()["p"]
            if n == 0 or p == 0:
                return None
            return math.log(n / p)
        finally:
            conn.close()

    @staticmethod
    def GetWordWeightInDocument(lemma_str, document_id):
        conn = db.get_connection()
        try:
            row = conn.execute(
                "SELECT it.frequency AS q FROM index_terms it"
                " JOIN lemmas l ON l.lemma_id = it.lemma_id"
                " WHERE l.lemma = ? AND it.document_id = ?",
                (lemma_str, document_id),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        b = Document.GetLemmInverseFrequency(lemma_str)
        if b is None:
            return None
        return row["q"] * b

    @staticmethod
    def GetLemmWeightInDocument(lemma_id, document_id):
        conn = db.get_connection()
        try:
            row = conn.execute(
                "SELECT frequency AS q FROM index_terms"
                " WHERE lemma_id = ? AND document_id = ?",
                (lemma_id, document_id),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        b = Document.GetLemmInverseFrequency(lemma_id)
        if b is None:
            return None
        return row["q"] * b

    @staticmethod
    def GetDocumentVector(document_id):
        conn = db.get_connection()
        try:
            rows = conn.execute(
                "SELECT lemma_id, frequency FROM index_terms WHERE document_id = ?",
                (document_id,),
            ).fetchall()
        finally:
            conn.close()
        vector = {}
        for r in rows:
            b = Document.GetLemmInverseFrequency(r["lemma_id"])
            if b is not None:
                vector[r["lemma_id"]] = r["frequency"] * b
        return vector

    @staticmethod
    def get_postings(lemma_str):
        conn = db.get_connection()
        try:
            rows = conn.execute(
                "SELECT it.document_id AS d, it.frequency AS q"
                " FROM lemmas l JOIN index_terms it ON it.lemma_id = l.lemma_id"
                " WHERE l.lemma = ?",
                (lemma_str,),
            ).fetchall()
            return {r["d"]: r["q"] for r in rows}
        finally:
            conn.close()

    @staticmethod
    def get_surface_forms(document_id):
        conn = db.get_connection()
        try:
            rows = conn.execute(
                "SELECT l.lemma FROM index_terms it"
                " JOIN lemmas l ON l.lemma_id = it.lemma_id"
                " WHERE it.document_id = ?",
                (document_id,),
            ).fetchall()
            return {r["lemma"] for r in rows}
        finally:
            conn.close()
