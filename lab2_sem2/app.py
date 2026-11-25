from __future__ import annotations

import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import (Flask, abort, flash, render_template, request,
                   send_from_directory, redirect, url_for, Response)

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from recognizer import (AlphabetProfile, FrequencyWordsProfile,
                        NeuralClassifier, html_to_text, letter_vector,
                        tokenize_words)
from recognizer.neural import make_feature_vector              

app = Flask(__name__)
app.secret_key = "lang-id-lab-v25"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  

UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
ALLOWED_EXT = {".html", ".htm", ".txt"}

METHOD_TITLES = {
    "freq": "Частотные слова",
    "alpha": "Алфавитный",
    "neural": "Нейросеть",
}

# Загрузка (или автоматическое построение) моделей
_freq_profiles: dict[str, FrequencyWordsProfile] = {}
_alpha_profiles: dict[str, AlphabetProfile] = {}
_neural: NeuralClassifier | None = None


def ensure_models() -> None:
    global _neural
    need_train = not (MODELS_DIR / "nn_model.joblib").exists()
    if need_train:
        print("Модели не найдены — запускается автоматическое обучение...")
        import train as train_module
        train_module.main()

    for lang in ("fr", "en"):
        _freq_profiles[lang] = FrequencyWordsProfile.load(
            MODELS_DIR / f"freq_{lang}.json")
        _alpha_profiles[lang] = AlphabetProfile.load(
            MODELS_DIR / f"alpha_{lang}.json")
    _neural = NeuralClassifier.load(MODELS_DIR / "nn_model.joblib",
                                    MODELS_DIR / "nn_words.json")
    print("Модели загружены.")


# Логика анализа одного документа
def guess_label(filename: str) -> str | None:
    name = filename.lower()
    if name.startswith("fr"):
        return "fr"
    if name.startswith("en"):
        return "en"
    return None


def analyze_text(filename: str, html: str) -> dict:
    """Полный анализ одного документа тремя методами (Шаги 2-3 алгоритма)."""
    t_start = time.perf_counter()

    # Шаг 1: предварительная обработка
    text = html_to_text(html)
    tokens = tokenize_words(text)

    methods = {}

    # Метод 1: частотные слова (максимум суммарной лог-вероятности)
    t0 = time.perf_counter()
    scores = {lang: p.score(tokens) for lang, p in _freq_profiles.items()}
    best = max(scores, key=scores.get)
    matched = _freq_profiles[best].matched_ratio(tokens)
    methods["freq"] = {
        "lang": best,
        "score": round(scores[best], 1),
        "detail": f"оценка {scores[best]:.1f} лог.ед./1000 слов, "
                  f"в ПОЯ {matched:.0%} слов",
        "ms": round((time.perf_counter() - t0) * 1000, 2),
    }

    # Метод 2: алфавитный (минимум расстояния L1)
    t0 = time.perf_counter()
    doc_vector = letter_vector(text)
    dists = {lang: p.distance(doc_vector) for lang, p in _alpha_profiles.items()}
    best = min(dists, key=dists.get)
    methods["alpha"] = {
        "lang": best,
        "score": round(dists[best], 4),
        "detail": "расстояние " + ", ".join(
            f"{lang}: {d:.4f}" for lang, d in sorted(dists.items())),
        "ms": round((time.perf_counter() - t0) * 1000, 2),
    }

    # Метод 3: нейросеть (MLP, вероятность класса)
    t0 = time.perf_counter()
    lang_pred, prob = _neural.predict(text)
    methods["neural"] = {
        "lang": lang_pred,
        "score": round(prob, 4),
        "detail": f"вероятность класса {prob:.2%}",
        "ms": round((time.perf_counter() - t0) * 1000, 2),
    }

    # Итоговое решение: большинство голосов 
    votes = [m["lang"] for m in methods.values()]
    final = max(set(votes), key=votes.count)
    label = guess_label(filename)

    return {
        "file": filename,
        "label": label,
        "final": final,
        "correct": None if label is None else (label == final),
        "chars": len(text),
        "words": len(tokens),
        "methods": methods,
        "total_ms": round((time.perf_counter() - t_start) * 1000, 2),
    }


def summarize(items: list[dict]) -> dict:
    labeled = [it for it in items if it["label"] is not None]

    def acc(method: str | None) -> str | None:
        if not labeled:
            return None
        if method is None:
            hits = sum(1 for it in labeled if it["correct"])
        else:
            hits = sum(1 for it in labeled if it["methods"][method]["lang"]
                       == it["label"])
        return f"{hits / len(labeled):.0%}"

    def avg_ms(method: str) -> float:
        vals = [it["methods"][method]["ms"] for it in items]
        return round(sum(vals) / max(len(vals), 1), 2)

    summary = {
        "total": len(items),
        "fr": sum(1 for it in items if it["final"] == "fr"),
        "en": sum(1 for it in items if it["final"] == "en"),
        "labeled": len(labeled),
        "acc_final": acc(None),
        "acc_freq": acc("freq"),
        "acc_alpha": acc("alpha"),
        "acc_neural": acc("neural"),
        "avg_ms": {m: avg_ms(m) for m in METHOD_TITLES},
        "avg_total_ms": round(
            sum(it["total_ms"] for it in items) / max(len(items), 1), 2),
    }
    return summary


def build_report_text(batch: dict) -> str:
    b = batch["summary"]
    lines = [
        "ОТЧЁТ ОБ АВТОМАТИЧЕСКОМ РАСПОЗНАВАНИИ ЯЗЫКА ТЕКСТА",
        f"Лабораторная работа, Вариант 25 (языки: французский, английский)",
        f"Дата: {batch['created']}",
        "=" * 78,
        "",
        f"Проанализировано документов: {b['total']}",
        f"Распознано как французский: {b['fr']}",
        f"Распознано как английский : {b['en']}",
        f"Точность (итоговая): {b['acc_final'] or '—'} "
        f"(по именам файлов: {b['labeled']} размеченных документов)",
        f"Точность по методам: частотные слова {b['acc_freq'] or '—'}, "
        f"алфавитный {b['acc_alpha'] or '—'}, "
        f"нейросеть {b['acc_neural'] or '—'}",
        f"Среднее время: {b['avg_total_ms']:.2f} мс/документ",
        "=" * 78,
        "",
    ]
    for it in batch["items"]:
        lines += [
            f"Документ: {it['file']}",
            f"  Объём: {it['chars']} символов, {it['words']} слов",
            f"  Метка из имени файла: "
            f"{it['label'].upper() if it['label'] else 'нет'}",
            f"  Метод частотных слов : {it['methods']['freq']['lang'].upper()} "
            f"({it['methods']['freq']['detail']})",
            f"  Алфавитный метод     : {it['methods']['alpha']['lang'].upper()} "
            f"({it['methods']['alpha']['detail']})",
            f"  Нейросетевой метод   : {it['methods']['neural']['lang'].upper()} "
            f"({it['methods']['neural']['detail']})",
            f"  ИТОГ: {it['final'].upper()} "
            f"{'— верно' if it['correct'] else '' if it['correct'] is None else '— ошибка'}",
            f"  Время обработки: {it['total_ms']:.2f} мс",
            "-" * 78,
        ]
    return "\n".join(lines)

# Хранилище результатов пакетов (в памяти + JSON на диске)
BATCHES: dict[str, dict] = {}


def load_batch(batch_id: str) -> dict | None:
    if batch_id in BATCHES:
        return BATCHES[batch_id]
    path = RESULTS_DIR / f"{batch_id}.json"
    if path.exists():
        batch = json.loads(path.read_text(encoding="utf-8"))
        BATCHES[batch_id] = batch
        return batch
    return None

# Маршруты

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    files = request.files.getlist("documents")
    if not files or all(not f.filename for f in files):
        flash("Не выбран ни один файл. Загрузите HTML-документы.")
        return redirect(url_for("index"))

    batch_id = uuid.uuid4().hex[:12]
    batch_dir = UPLOAD_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for f in files:
        if not f.filename:
            continue
        safe_name = re.sub(r"[^\w\.\- ]+", "_", f.filename).strip() or "doc"
        ext = Path(safe_name).suffix.lower()
        if ext not in ALLOWED_EXT:
            flash(f"Файл {f.filename} пропущен: допустимы HTML и TXT.")
            continue
        raw = f.read()
        if not raw.strip():
            continue
        (batch_dir / safe_name).write_bytes(raw)
        items.append(analyze_text(safe_name, raw.decode("utf-8", "replace")))

    if not items:
        flash("Не удалось обработать ни один документ.")
        return redirect(url_for("index"))

    batch = {
        "id": batch_id,
        "created": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
        "items": items,
        "summary": summarize(items),
    }
    BATCHES[batch_id] = batch
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{batch_id}.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")
    (RESULTS_DIR / f"{batch_id}.txt").write_text(
        build_report_text(batch), encoding="utf-8")
    return redirect(url_for("results", batch_id=batch_id))


@app.route("/results/<batch_id>")
def results(batch_id: str):
    batch = load_batch(batch_id)
    if batch is None:
        abort(404)
    return render_template("results.html", batch=batch,
                           method_titles=METHOD_TITLES)


@app.route("/doc/<batch_id>/<path:filename>")
def document(batch_id: str, filename: str):
    """Активная ссылка на исходный загруженный документ."""
    batch_dir = (UPLOAD_DIR / batch_id).resolve()
    if not batch_dir.exists():
        abort(404)
    return send_from_directory(batch_dir, filename)


@app.route("/doc/<batch_id>/<path:filename>/text")
def document_text(batch_id: str, filename: str):
    """Просмотр извлечённого текста документа."""
    path = (UPLOAD_DIR / batch_id / filename).resolve()
    if not path.exists():
        abort(404)
    text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    return render_template("doc_view.html", filename=filename,
                           text=text[:8000], batch_id=batch_id)


@app.route("/report/<batch_id>.txt")
def report(batch_id: str):
    """Сохранение результатов в текстовый файл (скачивание)."""
    path = RESULTS_DIR / f"{batch_id}.txt"
    if not path.exists():
        batch = load_batch(batch_id)
        if batch is None:
            abort(404)
        RESULTS_DIR.mkdir(exist_ok=True)
        path.write_text(build_report_text(batch), encoding="utf-8")
    return send_from_directory(RESULTS_DIR, f"{batch_id}.txt",
                               as_attachment=True,
                               download_name="language_report.txt")


@app.route("/help")
def help_page():
    meta_path = MODELS_DIR / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) \
        if meta_path.exists() else {}
    return render_template("help.html", meta=meta)


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", code=404,
                           message="Страница не найдена"), 404


@app.errorhandler(413)
def too_large(_e):
    flash("Суммарный размер загрузки превышает 32 МБ.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    ensure_models()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
