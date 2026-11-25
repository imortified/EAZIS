from __future__ import annotations

import json
from pathlib import Path

from .preprocessing import ALPHABET, letter_vector


class AlphabetProfile:

    def __init__(self, lang: str = "") -> None:
        self.lang = lang
        self.freqs: dict[str, float] = {c: 0.0 for c in ALPHABET}

    @classmethod
    def build(cls, lang: str, texts: list[str]) -> "AlphabetProfile":
        profile = cls(lang)
        joined = "\n".join(texts)
        profile.freqs = letter_vector(joined)
        return profile

    def distance(self, doc_vector: dict[str, float]) -> float:  # Манхэттенское расстояние между распределениями букв
        return sum(abs(self.freqs[c] - doc_vector.get(c, 0.0))
                   for c in ALPHABET)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "lang": self.lang,
            "method": "alphabet",
            "alphabet": ALPHABET,
            "freqs": self.freqs,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AlphabetProfile":
        data = json.loads(path.read_text(encoding="utf-8"))
        profile = cls(data["lang"])
        profile.freqs = {c: data["freqs"].get(c, 0.0) for c in ALPHABET}
        return profile
