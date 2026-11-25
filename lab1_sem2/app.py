import time

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, abort)

from ips import db
from ips.document import Document
from ips.collector import DocumentCollector
from ips.search import Search
from ips import metrics as ips_metrics
from ips.seed import seed_if_empty

app = Flask(__name__)
app.secret_key = "ips-variant25-key"

db.init_db()
seed_if_empty()

@app.route("/")
def index():
    return render_template("index.html", results=None, query="", mode="all",
                           date_from="", date_to="", error=None, unknown=[],
                           suggestions=[], suggestion_query=None)


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "all")
    date_from = request.args.get("date_from", "").strip() or None
    date_to = request.args.get("date_to", "").strip() or None

    s = Search(all_words_together=(mode == "all"),
               date_start=date_from, date_end=date_to)
    s.search_query = query
    results = s.GetSearchResult() if query else []

    return render_template("index.html", results=results, query=query,
                           mode=mode, date_from=date_from or "",
                           date_to=date_to or "",
                           error=s.last_error, unknown=s.unknown_terms,
                           suggestions=s.suggestions,
                           suggestion_query=s.suggestion_query or None)


@app.route("/document/<int:doc_id>")
def document_view(doc_id):
    row = db.get_document_row(doc_id)
    if row is None:
        abort(404)
    words = len(row["text"].split())
    return render_template("document_view.html", doc=row, words=words)


@app.route("/documents")
def documents_page():
    docs = db.all_documents()
    return render_template("documents.html", docs=docs,
                           n_docs=db.total_documents(),
                           n_lemmas=db.total_lemmas())


@app.route("/documents/add", methods=["POST"])
def documents_add():
    title = request.form.get("title", "").strip()
    text = request.form.get("text", "").strip()
    if not title or not text:
        flash("Заполните заголовок и текст документа.", "error")
    else:
        doc = Document(title=title, text=text)
        if doc.AddDocumentToBase():
            flash(f"Документ №{doc.document_id} добавлен и проиндексирован.", "ok")
        else:
            flash("Не удалось проиндексировать документ.", "error")
    return redirect(url_for("documents_page"))


@app.route("/documents/upload", methods=["POST"])
def documents_upload():
    file = request.files.get("file")
    if file is None or file.filename == "":
        flash("Выберите файл .txt или .md.", "error")
        return redirect(url_for("documents_page"))
    collector = DocumentCollector()
    doc_id, err = collector.add_uploaded_file(file.filename, file.read())
    if err:
        flash(err, "error")
    else:
        flash(f"Файл проиндексирован: документ №{doc_id}.", "ok")
    return redirect(url_for("documents_page"))


@app.route("/documents/folder", methods=["POST"])
def documents_folder():
    """Модуль отбора документов: индексация каталога (в т.ч. сетевой папки ЛВС)."""
    folder = request.form.get("folder", "")
    collector = DocumentCollector()
    stats = collector.collect_from_folder(folder)
    if stats["errors"]:
        for e in stats["errors"]:
            flash(e, "error")
    if stats["scanned"] == 0 and not stats["errors"]:
        flash("В папке не найдено файлов .txt/.md.", "error")
    else:
        flash(f"Отобрано файлов: {stats['scanned']}, добавлено: {stats['added']},"
              f" пропущено: {stats['skipped']}.", "ok")
    return redirect(url_for("documents_page"))


@app.route("/documents/<int:doc_id>/delete", methods=["POST"])
def documents_delete(doc_id):
    if Document.DeleteDocumentFromBase(doc_id):
        flash(f"Документ №{doc_id} удалён из базы.", "ok")
    else:
        flash("Документ не найден.", "error")
    return redirect(url_for("documents_page"))


@app.route("/metrics")
def metrics_page():
    data = ips_metrics.evaluate_all()
    charts = ips_metrics.render_charts(data)
    docs = db.all_documents()
    stamp = int(time.time())
    return render_template("metrics.html", data=data, charts=charts,
                           docs=docs, v=stamp)


@app.route("/metrics/query", methods=["POST"])
def metrics_add_query():
    query_text = request.form.get("query_text", "").strip()
    description = request.form.get("description", "").strip()
    relevant = [int(v) for v in request.form.getlist("rel")]
    if not query_text:
        flash("Введите текст тестового запроса.", "error")
    elif not relevant:
        flash("Отметьте хотя бы один релевантный документ.", "error")
    else:
        db.add_test_query(query_text, description, relevant)
        flash("Тестовый запрос добавлен. Метрики пересчитаны.", "ok")
    return redirect(url_for("metrics_page"))


@app.route("/metrics/query/<int:query_id>/delete", methods=["POST"])
def metrics_delete_query(query_id):
    db.delete_test_query(query_id)
    flash("Тестовый запрос удалён.", "ok")
    return redirect(url_for("metrics_page"))


@app.route("/help")
def help_page():
    return render_template("help.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
