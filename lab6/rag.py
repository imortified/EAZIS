import json
import numpy as np
from database import get_db
from nlp import lemmatize_text, get_embedding, split_text


def store_document(filename, text):
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO documents (filename, content) VALUES (?, ?)",
        (filename, text)
    )
    doc_id = cursor.lastrowid

    chunks = split_text(text)
    for i, chunk in enumerate(chunks):
        # Лемматизируем чанк для построения эмбеддинга
        lemma_chunk = lemmatize_text(chunk)
        # Эмбеддинг строим от лемматизированного текста — улучшает поиск
        emb = get_embedding(lemma_chunk)
        emb_json = json.dumps(emb.tolist())
        cursor.execute(
            "INSERT INTO chunks (document_id, content, lemma_content, embedding, chunk_index) VALUES (?, ?, ?, ?, ?)",
            (doc_id, chunk, lemma_chunk, emb_json, i)
        )

    db.commit()
    print(f"[INFO] Документ сохранён: id={doc_id}, чанков={len(chunks)} (с лемматизацией)")
    return doc_id, len(chunks)


def search_chunks(query, top_k=4): # Поиск релевантных чанков по косинусному сходству с лемматизацией запроса
    db = get_db()
    lemma_query = lemmatize_text(query)
    print(f"[DEBUG] Оригинальный запрос: '{query}'")
    print(f"[DEBUG] Лемматизированный: '{lemma_query}'")

    query_emb = get_embedding(lemma_query)
    cursor = db.execute("SELECT id, content, lemma_content, embedding FROM chunks")
    rows = cursor.fetchall()

    results = []
    for row in rows:
        chunk_emb = np.array(json.loads(row['embedding']))
        dot = np.dot(query_emb, chunk_emb)
        norm = np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)
        similarity = dot / norm if norm > 0 else 0
        results.append((similarity, row['content'], row['lemma_content']))

    results.sort(reverse=True, key=lambda x: x[0])
    top = results[:top_k]
    for sim, orig, lemma in top:
        print(f"[DEBUG] chunk sim={sim:.3f} | lemma='{lemma[:60]}...'")

    return [r[1] for r in top]
