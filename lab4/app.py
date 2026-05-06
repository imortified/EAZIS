import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename
from database import init_db, save_analysis, get_all_analyses, get_analysis, update_analysis_text, delete_analysis
from text_extractor import extract_text
from analyzer import analyze_text

app = Flask(__name__)
app.secret_key = "semantic-syntax-lab4-secret-key"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 * 64

ALLOWED_EXTENSIONS = {"txt", "rtf", "pdf", "html", "doc", "docx"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("Файл не выбран", "error")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("Файл не выбран", "error")
        return redirect(url_for("index"))

    ext = file.filename.rsplit(".", 1)[1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        flash(f"Неподдерживаемый формат: {ext}", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    raw_text = extract_text(filepath)
    if raw_text is None or raw_text.strip() == "":
        flash("Не удалось извлечь текст из файла", "error")
        return redirect(url_for("index"))

    try:
        result = analyze_text(raw_text)
    except Exception as e:
        flash(f"Ошибка анализа: {str(e)}", "error")
        return redirect(url_for("index"))

    analysis_id = save_analysis(filename, raw_text, result["sentences"])
    flash("Анализ успешно выполнен", "success")
    return redirect(url_for("analysis", id=analysis_id))

@app.route("/analysis/<int:id>")
def analysis(id):
    data = get_analysis(id)
    if not data:
        flash("Анализ не найден", "error")
        return redirect(url_for("history"))
    return render_template("analysis.html", data=data)

@app.route("/history")
def history():
    analyses = get_all_analyses()
    return render_template("history.html", analyses=analyses)

@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit(id):
    data = get_analysis(id)
    if not data:
        flash("Анализ не найден", "error")
        return redirect(url_for("history"))

    if request.method == "POST":
        new_text = request.form.get("text", "")
        update_analysis_text(id, new_text)
        flash("Текст обновлен", "success")
        return redirect(url_for("analysis", id=id))

    return render_template("edit.html", data=data)

@app.route("/delete/<int:id>", methods=["POST"])
def delete(id):
    delete_analysis(id)
    flash("Анализ удален", "success")
    return redirect(url_for("history"))

@app.route("/export/<int:id>")
def export(id):
    data = get_analysis(id)
    if not data:
        flash("Анализ не найден", "error")
        return redirect(url_for("history"))

    export_path = os.path.join(app.config["UPLOAD_FOLDER"], f"analysis_{id}.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return send_file(export_path, as_attachment=True, download_name=f"analysis_{id}.json")

@app.route("/help")
def help_page():
    return render_template("help.html")

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
