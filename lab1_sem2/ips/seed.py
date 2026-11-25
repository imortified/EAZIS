from . import db
from .document import Document
from .seed_texts import DEMO_DOCS, TEST_QUERIES


def seed_if_empty():
    """Заполнить базу при первом запуске. Возвращает True, если база заполнена."""
    if db.total_documents() > 0:
        return False

    for doc in DEMO_DOCS:
        Document(
            title=doc["title"],
            text=doc["text"],
            date=doc["date"],
            time=doc["time"],
            source="демонстрационная коллекция",
        ).AddDocumentToBase()

    for query_text, description, relevant in TEST_QUERIES:
        # document_id в TEST_QUERIES идут в порядке добавления демо-документов
        db.add_test_query(query_text, description, sorted(relevant))
    return True
