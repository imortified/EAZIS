# -*- coding: utf-8 -*-
"""
Построение профилей языков и обучение нейросети (Шаг 2 алгоритма).

Использование:
    python train.py

Создаёт в папке models/:
    freq_fr.json, freq_en.json   — профили метода частотных слов;
    alpha_fr.json, alpha_en.json — профили алфавитного метода;
    nn_model.joblib, nn_words.json — нейросеть и список признаковых слов;
    meta.json                    — сведения о моделях.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from recognizer import FrequencyWordsProfile, AlphabetProfile, NeuralClassifier  # noqa: E402
from recognizer.neural import split_chunks                                      # noqa: E402

CORPUS_DIR = BASE_DIR / "corpus" / "train"
MODELS_DIR = BASE_DIR / "models"
LANGS = ("fr", "en")
TOP_WORDS_FOR_NN = 30        # слов от каждого языка в признаках нейросети


def load_training_texts(lang: str) -> list[str]:
    """Прочитать все HTML-файлы тренировочного корпуса и извлечь текст."""
    from recognizer.preprocessing import html_to_text

    texts = []
    folder = CORPUS_DIR / lang
    for path in sorted(folder.glob("*.html")):
        texts.append(html_to_text(path.read_text(encoding="utf-8")))
    return texts


def build_word_list(freq_profiles: dict[str, FrequencyWordsProfile]) -> list[str]:
    """Объединённый список частотных слов двух языков для признаков нейросети."""
    merged: dict[str, float] = {}
    for profile in freq_profiles.values():
        for i, (word, prob) in enumerate(profile.words.items()):
            if i >= TOP_WORDS_FOR_NN:
                break
            merged[word] = max(merged.get(word, 0.0), prob)
    # порядок: более частые слова первыми
    return [w for w, _ in sorted(merged.items(), key=lambda kv: -kv[1])]


def main() -> None:
    started = time.time()
    MODELS_DIR.mkdir(exist_ok=True)

    print("1) Чтение тренировочного корпуса")
    train_texts = {lang: load_training_texts(lang) for lang in LANGS}
    for lang in LANGS:
        size = sum(len(t) for t in train_texts[lang])
        print(f"   {lang.upper()}: {len(train_texts[lang])} файлов, {size} символов")

    print("2) Профили метода частотных слов")
    freq_profiles = {}
    for lang in LANGS:
        profile = FrequencyWordsProfile.build(lang, train_texts[lang])
        profile.save(MODELS_DIR / f"freq_{lang}.json")
        freq_profiles[lang] = profile
        top5 = list(profile.words.items())[:5]
        print(f"   {lang.upper()}: топ-5: {top5}")

    print("3) Профили алфавитного метода")
    alpha_profiles = {}
    for lang in LANGS:
        profile = AlphabetProfile.build(lang, train_texts[lang])
        profile.save(MODELS_DIR / f"alpha_{lang}.json")
        alpha_profiles[lang] = profile
        notable = {c: round(p, 4) for c, p in profile.freqs.items() if p > 0.02}
        print(f"   {lang.upper()}: заметные буквы: {notable}")

    print("4) Нейросетевой метод (MLP)")
    word_list = build_word_list(freq_profiles)
    samples = []
    for lang in LANGS:
        for text in train_texts[lang]:
            for chunk in split_chunks(text):
                samples.append((chunk, lang))
    print(f"   фрагментов для обучения: {len(samples)} "
          f"(признаков: {43 + len(word_list)})")

    neural = NeuralClassifier(word_list)
    info = neural.train(samples)
    neural.save(MODELS_DIR / "nn_model.joblib", MODELS_DIR / "nn_words.json")
    print(f"   точность на обучении: {info['train_accuracy']:.2%}, "
          f"итераций: {info['n_iterations']}")

    meta = {
        "languages": list(LANGS),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_chars": {lang: sum(len(t) for t in train_texts[lang])
                        for lang in LANGS},
        "neural": info,
        "word_list_size": len(word_list),
    }
    (MODELS_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Готово за {time.time() - started:.1f} с. Модели: {MODELS_DIR}")


if __name__ == "__main__":
    main()
