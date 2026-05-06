import os
from flask import Flask, render_template, request, jsonify
from config import BASE_DIR, UPLOAD_FOLDER, MAX_CONTENT_LENGTH, DATABASE_PATH
from database import init_db, close_db
from nlp import extract_text_from_pdf
from rag import store_document, search_chunks
from llm import generate_response

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.teardown_appcontext
def teardown(exception):
    close_db()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/conversations', methods=['GET', 'POST'])
def conversations():
    from database import get_db
    db = get_db()
    if request.method == 'POST':
        title = request.json.get('title', 'Новый диалог')
        cursor = db.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
        db.commit()
        print(f"[INFO] Создан диалог #{cursor.lastrowid}: {title}")
        return jsonify({'id': cursor.lastrowid, 'title': title})
    else:
        rows = db.execute("SELECT * FROM conversations ORDER BY created_at DESC").fetchall()
        return jsonify([dict(row) for row in rows])


@app.route('/api/conversations/<int:conv_id>', methods=['GET', 'DELETE', 'PUT'])
def conversation(conv_id):
    from database import get_db
    db = get_db()
    if request.method == 'GET':
        rows = db.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp",
            (conv_id,)
        ).fetchall()
        return jsonify([dict(row) for row in rows])
    elif request.method == 'DELETE':
        db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        db.commit()
        print(f"[INFO] Удалён диалог #{conv_id}")
        return jsonify({'status': 'deleted'})
    elif request.method == 'PUT':
        title = request.json.get('title')
        db.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
        db.commit()
        return jsonify({'status': 'updated'})


@app.route('/api/conversations/<int:conv_id>/messages', methods=['POST'])
def add_message(conv_id):
    from database import get_db
    db = get_db()
    data = request.json
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'error': 'Пустое сообщение'}), 400

    db.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conv_id, 'user', user_msg)
    )
    db.commit()
    print(f"[INFO] Сообщение пользователя сохранено в диалог #{conv_id}")

    history_rows = db.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp",
        (conv_id,)
    ).fetchall()
    history = [dict(row) for row in history_rows]

    # RAG: поиск с лемматизацией через pymorphy3
    context_chunks = search_chunks(user_msg, top_k=4)
    context = "\n---\n".join(context_chunks) if context_chunks else "Контекст отсутствует."
    response_text = generate_response(user_msg, context, history)

    db.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conv_id, 'assistant', response_text)
    )
    db.commit()
    print(f"[INFO] Ответ ассистента сохранён в диалог #{conv_id}")
    return jsonify({'response': response_text})


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Имя файла пустое'}), 400
    if file and file.filename.lower().endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        text = extract_text_from_pdf(filepath)
        if not text.strip():
            return jsonify({'error': 'Не удалось извлечь текст из PDF'}), 400
        doc_id, chunks_count = store_document(file.filename, text)
        return jsonify({
            'status': 'success',
            'document_id': doc_id,
            'chunks': chunks_count,
            'filename': file.filename
        })
    return jsonify({'error': 'Требуется PDF'}), 400


@app.route('/api/documents', methods=['GET'])
def list_documents():
    from database import get_db
    db = get_db()
    rows = db.execute(
        "SELECT id, filename, created_at FROM documents ORDER BY created_at DESC"
    ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    from database import get_db
    db = get_db()
    db.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
    db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    db.commit()
    return jsonify({'status': 'deleted'})


if __name__ == '__main__':
    init_db()
    print(f"[INFO] БД: {DATABASE_PATH}")
    print("[INFO] Запуск сервера на http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
