from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import letter_vector, tokenize_words

FEATURE_LETTERS = 43          # буквенных признаков (единый алфавит)
CHUNK_SIZE = 1200             # размер фрагмента при нарезке корпуса, символов


def make_feature_vector(text: str, word_list: list[str]) -> list[float]:
    """Вектор признаков документа: частоты букв + частоты слов списка."""
    letters = letter_vector(text)
    tokens = tokenize_words(text)
    total = max(len(tokens), 1)
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    word_freqs = [counts.get(w, 0) / total for w in word_list]
    return [letters[c] for c in letters] + word_freqs


def split_chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Нарезать текст на фрагменты ~size символов по границам абзацев."""
    chunks, current = [], ""
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        if len(current) + len(block) + 1 > size and current:
            chunks.append(current)
            current = block
        else:
            current = f"{current} {block}".strip()
    if len(current) >= size // 3:            # слишком короткий хвост отбрасываем
        chunks.append(current)
    return chunks


class NeuralClassifier:
    """Обёртка над Pipeline(StandardScaler, MLPClassifier)."""

    def __init__(self, word_list: list[str] | None = None) -> None:
        self.word_list: list[str] = word_list or []
        self.pipeline: Pipeline | None = None

    def train(self, samples: list[tuple[str, str]]) -> dict:
        """samples — пары (текст_фрагмента, метка_языка)."""
        X = np.array([make_feature_vector(t, self.word_list) for t, _ in samples])
        y = np.array([lang for _, lang in samples])

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(24,),
                activation="relu",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=42,
            )),
        ])
        self.pipeline.fit(X, y)

        train_acc = self.pipeline.score(X, y)
        info = {
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "train_accuracy": round(float(train_acc), 4),
            "n_iterations": int(self.pipeline.named_steps["mlp"].n_iter_),
            "classes": [str(c) for c in self.pipeline.named_steps["mlp"].classes_],
        }
        return info

    # --------------------------- классификация -----------------------------
    def predict(self, text: str) -> tuple[str, float]:
        """Вернуть (язык, вероятность победившего класса)."""
        X = np.array([make_feature_vector(text, self.word_list)])
        proba = self.pipeline.predict_proba(X)[0]
        classes = self.pipeline.named_steps["mlp"].classes_
        best = int(np.argmax(proba))
        return str(classes[best]), float(proba[best])

    def save(self, model_path: Path, words_path: Path) -> None:
        joblib.dump(self.pipeline, model_path)
        words_path.write_text(json.dumps(self.word_list, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, model_path: Path, words_path: Path) -> "NeuralClassifier":
        classifier = cls()
        classifier.word_list = json.loads(words_path.read_text(encoding="utf-8"))
        classifier.pipeline = joblib.load(model_path)
        return classifier
