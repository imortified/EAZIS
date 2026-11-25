from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

TOP_N = 200           


class FrequencyWordsProfile:

    def __init__(self, lang: str = "") -> None:
        self.lang = lang
        self.words: dict[str, float] = {}   # слово -> вероятность в языке
        self.min_prob: float = 0.0          # вероятность отсутствующих слов

    @classmethod
    def build(cls, lang: str, texts: list[str], top_n: int = TOP_N) -> "FrequencyWordsProfile":
        from .preprocessing import tokenize_words

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize_words(text))

        total = sum(counter.values())
        profile = cls(lang)
        # вероятности слов ПОЯ — доли от общего числа слов корпуса
        for word, count in counter.most_common(top_n):
            profile.words[word] = count / total
        # минимальная вероятность: редчайшее слово ПОЯ / 2
        rarest = min(profile.words.values()) if profile.words else 1.0 / max(total, 1)
        profile.min_prob = rarest / 2.0
        return profile

    def score(self, tokens: list[str]) -> float:     # оценка документа
        if not tokens:
            return 0.0
        total = 0.0
        for token in tokens:
            p = self.words.get(token, self.min_prob)
            total += math.log(p)
        return total / len(tokens) * 1000.0

    def matched_ratio(self, tokens: list[str]) -> float: # Доля слов документа, найденных в ПОЯ
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens if t in self.words)
        return hits / len(tokens)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "lang": self.lang,
            "method": "frequency_words",
            "top_n": TOP_N,
            "min_prob": self.min_prob,
            "words": self.words,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FrequencyWordsProfile":
        data = json.loads(path.read_text(encoding="utf-8"))
        profile = cls(data["lang"])
        profile.words = data["words"]
        profile.min_prob = data["min_prob"]
        return profile
