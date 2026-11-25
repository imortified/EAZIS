from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup

# Единый алфавит признаков: 26 базовых латинских букв + диакритика,
# характерная для французского языка, + лигатуры oe, ae.
BASE_LETTERS = "abcdefghijklmnopqrstuvwxyz"
ACCENTED = "àâäçéèêëîïôöùûüÿœæ"
ALPHABET = [c for c in BASE_LETTERS] + [c for c in ACCENTED]

# Токен: последовательность букв единого алфавита (без цифр и пунктуации).
_TOKEN_RE = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœæ]+")


def html_to_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:                     
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "head", "title",
                     "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return normalize(text)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def tokenize_words(text: str) -> list[str]:
    text = text.replace("’", "'").replace("ʼ", "'")
    text = re.sub(r"['`´]", " ", text)
    return _TOKEN_RE.findall(text)


def letter_vector(text: str) -> dict[str, float]:
    counts = {c: 0 for c in ALPHABET}
    total = 0
    for ch in text:
        if ch in counts:
            counts[ch] += 1
            total += 1
    if total == 0:
        return {c: 0.0 for c in ALPHABET}
    return {c: counts[c] / total for c in ALPHABET}
