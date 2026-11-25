import re
from dataclasses import dataclass

from . import db
from .document import tokenize, STOPWORDS
from .stemmer import stem_word

__all__ = [
    "Correction", "damerau_levenshtein", "max_edit_distance",
    "build_dictionary", "suggest_correction", "suggest_query_fix",
]


def damerau_levenshtein(a, b, max_dist=None):     # Расстояние Дамерау–Левенштейна между строками
    la, lb = len(a), len(b)
    if max_dist is not None and abs(la - lb) > max_dist:
        return max_dist + 1
    if la == 0:
        return lb
    if lb == 0:
        return la

    prev2 = None                  # строка DP для i-2 (нужна для транспозиций)
    prev = list(range(lb + 1))    # строка DP для i-1
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d = min(prev[j] + 1,          # удаление
                    cur[j - 1] + 1,       # вставка
                    prev[j - 1] + cost)   # замена
            if (i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                d = min(d, prev2[j - 2] + 1)   # транспозиция соседних
            cur[j] = d
            if d < row_min:
                row_min = d
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev2, prev = prev, cur
    return prev[lb]


def max_edit_distance(length):
    if length <= 3:
        return 1
    if length <= 8:
        return 2
    return 3


@dataclass
class Correction:
    original: str      # слово, как его ввёл пользователь
    corrected: str     # предлагаемая словоформа из коллекции
    distance: int      # расстояние Дамерау–Левенштейна (0 для раскладки)
    kind: str          # "typo" — опечатка, "layout" — неверная раскладка


_DICT_CACHE = {"key": None, "stems": {}, "surface": {}, "forms": set()}

_LAYOUT_EN = "qwertyuiop[]asdfghjkl;'zxcvbnm,."
_LAYOUT_RU = "йцукенгшщзхъфывапролджэячсмитьбю"
_LAYOUT_EN_RU = str.maketrans(_LAYOUT_EN, _LAYOUT_RU)

OPERATOR_WORDS = {"and", "or", "not", "и", "или", "не"}

_TOKEN_PART_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)*", re.IGNORECASE)


def build_dictionary(force=False):
    conn = db.get_connection()
    try:
        n_docs = conn.execute(
            "SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        max_id = conn.execute(
            "SELECT COALESCE(MAX(document_id), 0) AS m FROM documents"
        ).fetchone()["m"]
        key = (n_docs, max_id)
        if not force and _DICT_CACHE["key"] == key:
            return _DICT_CACHE["stems"], _DICT_CACHE["surface"], _DICT_CACHE["forms"]

        # справочник основ: основа -> число документов с этой основой
        stems = {r["lemma"]: r["df"] for r in conn.execute(
            "SELECT l.lemma AS lemma, COUNT(it.document_id) AS df"
            " FROM lemmas l JOIN index_terms it ON it.lemma_id = l.lemma_id"
            " GROUP BY l.lemma")}

        # словоформы из исходных текстов: основа -> {форма: частота}
        form_freq = {}
        for row in conn.execute("SELECT text FROM documents"):
            for token in tokenize(row["text"]):
                if len(token) < 2 or token in STOPWORDS:
                    continue
                st = stem_word(token)
                forms = form_freq.setdefault(st, {})
                forms[token] = forms.get(token, 0) + 1
        # для показа выбираем самую частую форму (при равенстве — короткую)
        surface = {}
        all_forms = set()
        for st, forms in form_freq.items():
            all_forms.update(forms)
            surface[st] = sorted(
                forms.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0])
            )[0][0]

        _DICT_CACHE.update(key=key, stems=stems, surface=surface,
                           forms=all_forms)
        return stems, surface, all_forms
    finally:
        conn.close()

def _closest_stem(stem, stems):
    limit = max_edit_distance(len(stem))
    best, best_key = None, None
    for cand, df in stems.items():
        if abs(len(cand) - len(stem)) > limit:
            continue                     # отсев по длине
        d = damerau_levenshtein(stem, cand, limit)
        if d > limit:
            continue
        key = (d, -df, len(cand))
        if best_key is None or key < best_key:
            best, best_key = cand, key
    return best, (best_key[0] if best_key else -1)


def _layout_to_russian(word):
    low = (word or "").lower()
    if not low or not low.isascii() or not low.isalpha():
        return None
    ru = low.translate(_LAYOUT_EN_RU)
    return ru if ru != low else None


def _suggest_word(word, stems, surface, forms):
    if len(word) < 2 or word in STOPWORDS:
        return None
    if word in forms:
        return None                       # слово есть в коллекции
    st = stem_word(word)
    if st in stems:
        return None                       # основа знакома индексу

    cand, dist = _closest_stem(st, stems)
    if cand is not None:
        return Correction(word, surface.get(cand, cand), dist, "typo")

    ru = _layout_to_russian(word)
    if ru:
        # слово в правильной русской раскладке без опечаток
        rst = stem_word(ru)
        if rst in stems or ru in forms:
            return Correction(word, surface.get(rst, rst), 0, "layout")
        # раскладка + опечатка одновременно
        cand2, dist2 = _closest_stem(rst, stems)
        if cand2 is not None:
            return Correction(word, surface.get(cand2, cand2), dist2, "layout")
    return None


def suggest_correction(word):
    stems, surface, forms = build_dictionary()
    return _suggest_word((word or "").lower().strip(), stems, surface, forms)


def _replace_token(query, raw, new_raw):
    pat = re.compile(r"(?<![\w-])" + re.escape(raw) + r"(?![\w-])",
                     re.IGNORECASE)
    return pat.sub(new_raw, query)


def suggest_query_fix(query, stems=None, surface=None, forms=None):
    if stems is None or surface is None or forms is None:
        stems, surface, forms = build_dictionary()
    if not (query or "").strip() or not stems:
        return None, []

    corrected = query
    fixes = []
    seen = set()
    for match in _TOKEN_PART_RE.finditer(query):
        raw = match.group(0)
        low = raw.lower()
        if low in seen or low in OPERATOR_WORDS:
            continue
        seen.add(low)

        fixed_parts, token_fixes = [], []
        for part in low.split("-"):
            fix = _suggest_word(part, stems, surface, forms)
            if fix is not None:
                fixed_parts.append(fix.corrected.lower())
                token_fixes.append(fix)
            else:
                fixed_parts.append(part)
        if token_fixes:
            corrected = _replace_token(corrected, raw, "-".join(fixed_parts))
            fixes.extend(token_fixes)

    return (corrected, fixes) if fixes else (None, [])
