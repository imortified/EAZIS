import hashlib
import os

from . import db
from .document import Document

ALLOWED_EXT = (".txt", ".md")
MAX_FILE_BYTES = 2 * 1024 * 1024   # 2 МБ


def read_text_auto(raw_bytes):
    for enc in ("utf-8-sig", "utf-8", "cp1251", "utf-16"):
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def make_title(text, fallback):
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:100]
    return fallback[:100]


def _is_duplicate(text_hash):
    conn = db.get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE source LIKE ? LIMIT 1",
            (f"%{text_hash}%",),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


class DocumentCollector:
    def __init__(self):
        self.last_log = []

    def collect_from_folder(self, folder):
        stats = {"scanned": 0, "added": 0, "skipped": 0, "errors": []}
        self.last_log = []
        folder = folder.strip()

        if not folder:
            stats["errors"].append("Укажите путь к папке.")
            return stats
        if not os.path.isdir(folder):
            stats["errors"].append(f"Папка не найдена: {folder}")
            return stats

        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                ext = os.path.splitext(name)[1].lower()
                path = os.path.join(root, name)
                if ext not in ALLOWED_EXT:
                    continue
                stats["scanned"] += 1
                try:
                    if os.path.getsize(path) > MAX_FILE_BYTES:
                        stats["skipped"] += 1
                        self.last_log.append(f"пропущен (слишком большой): {path}")
                        continue
                    with open(path, "rb") as fh:
                        raw = fh.read()
                    text = read_text_auto(raw).strip()
                    if not text:
                        stats["skipped"] += 1
                        self.last_log.append(f"пропущен (пустой): {path}")
                        continue
                    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
                    if _is_duplicate(text_hash):
                        stats["skipped"] += 1
                        self.last_log.append(f"пропущен (дубликат): {path}")
                        continue

                    doc = Document(
                        title=make_title(text, os.path.splitext(name)[0]),
                        text=text,
                        source=f"отобран из папки; файл {name}; контрольная сумма {text_hash}",
                    )
                    if doc.AddDocumentToBase():
                        stats["added"] += 1
                        self.last_log.append(f"добавлен: {path}")
                    else:
                        stats["skipped"] += 1
                        self.last_log.append(f"не проиндексирован: {path}")
                except OSError as exc:
                    stats["errors"].append(f"{path}: {exc}")
        return stats

    def add_uploaded_file(self, filename, raw_bytes):
        name = os.path.basename(filename or "document.txt")
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXT:
            return None, "Поддерживаются только файлы .txt и .md"
        if len(raw_bytes) > MAX_FILE_BYTES:
            return None, "Файл больше 2 МБ"
        text = read_text_auto(raw_bytes).strip()
        if not text:
            return None, "Файл пустой"
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        if _is_duplicate(text_hash):
            return None, "Такой документ уже есть в базе (дубликат)"

        doc = Document(
            title=make_title(text, os.path.splitext(name)[0]),
            text=text,
            source=f"загружен файлом; имя {name}; контрольная сумма {text_hash}",
        )
        doc_id = doc.AddDocumentToBase()
        if doc_id is None:
            return None, "Не удалось проиндексировать документ"
        return doc_id, None
