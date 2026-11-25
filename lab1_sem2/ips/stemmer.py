VOWELS = set("аеиоуыэюя")

# Суффиксы, разбитые по группам (применяются по стадиям, как в Snowball)
PERFECTIVE_1 = ("ив", "ивши", "ившись", "ыв", "ывши", "ывшись")   # без ограничений
PERFECTIVE_2 = ("в", "вши", "вшись")                               # только после а/я
REFLEXIVE = ("ся", "сь")
ADJECTIVE = ("ее", "ие", "ые", "ое", "ими", "ыми", "ей", "ий", "ый", "ой",
             "ем", "им", "ым", "ом", "его", "ого", "ему", "ому", "их", "ых",
             "ую", "юю", "ая", "яя", "ою", "ею")
PARTICIPLE_1 = ("ивш", "ывш", "ующ")          # без ограничений
PARTICIPLE_2 = ("ем", "нн", "вш", "ющ", "щ")  # после а/я для вш/ющ/щ
VERB_1 = ("ла", "на", "ете", "йте", "ли", "й", "л", "ем", "н", "ло", "но",
          "ет", "ют", "ны", "ть", "ешь", "нно")                     # после а/я
VERB_2 = ("ила", "ыла", "ена", "ейте", "уйте", "ите", "или", "ыли", "ей",
          "уй", "ил", "ыл", "им", "ым", "ен", "ило", "ыло", "ено", "ят",
          "ует", "уют", "ит", "ыт", "ены", "ить", "ыть", "ишь", "ую", "ю")
NOUN = ("а", "ев", "ов", "ие", "ье", "е", "иями", "ями", "ами", "еи", "ии",
        "и", "ией", "ей", "ой", "ий", "й", "иям", "ям", "ием", "ем", "ам",
        "ом", "о", "у", "ах", "иях", "ях", "ы", "ь", "ию", "ью", "ю",
        "ия", "ья", "яя", "я",
        "s")  
DERIVATIONAL = ("ость", "ост")

MIN_WORD = 4        # слова короче не обрабатываем
MIN_STEM = 2        # минимальная длина основы


def _strip_longest(word, suffixes, require_last=None, min_stem=MIN_STEM):
    best = None
    for suf in suffixes:
        if word.endswith(suf) and (best is None or len(suf) > len(best)):
            best = suf
    if not best:
        return word
    base = word[: -len(best)]
    if len(base) < min_stem:
        return word
    if require_last and (not base or base[-1] not in require_last):
        return word
    return base


def _has_vowel(word):
    return any(ch in VOWELS for ch in word)


class RussianStemmer:

    def __init__(self):
        self._cache = {}

    def stem(self, word):
        w = word.lower().replace("ё", "е")
        if w in self._cache:
            return self._cache[w]
        result = self._stem(w)
        self._cache[w] = result
        return result

    def _stem(self, w):
        if len(w) <= MIN_WORD - 1:
            return w

        # 1. Деепричастия совершенного вида
        base = _strip_longest(w, PERFECTIVE_1, min_stem=3)
        if base == w:
            base = _strip_longest(w, PERFECTIVE_2, require_last="ая", min_stem=3)
        if base != w:
            w = base

        # 2. Возвратный суффикс
        w = _strip_longest(w, REFLEXIVE, min_stem=3)

        # 3. Прилагательное (+ причастие)
        base = _strip_longest(w, ADJECTIVE, min_stem=3)
        if base != w:
            w = base
            b2 = _strip_longest(w, PARTICIPLE_1, min_stem=4)
            if b2 == w:
                b2 = _strip_longest(w, PARTICIPLE_2, require_last="ая", min_stem=4)
            # снимаем также одиночное «н» после прилагательного: системный -> систем
            if b2 == w and w.endswith("н") and len(w) - 1 >= MIN_STEM:
                b2 = w[:-1]
            if b2 != w and _has_vowel(b2):
                w = b2

        # 4. Глагол
        base = _strip_longest(w, VERB_2, min_stem=3)
        if base == w:
            base = _strip_longest(w, VERB_1, require_last="ая", min_stem=3)
        if base != w and _has_vowel(base):
            w = base

        # 5. Существительное
        base = _strip_longest(w, NOUN, min_stem=3)
        if base != w and _has_vowel(base):
            w = base

        # 6. Словообразовательный суффикс
        base = _strip_longest(w, DERIVATIONAL, min_stem=4)
        if base != w:
            w = base

        # 7. Финальные упрощения
        if w.endswith("нн") and len(w) - 1 >= MIN_STEM:
            w = w[:-1]
        for suf in ("ейше", "ейш"):
            if w.endswith(suf) and len(w) - len(suf) >= MIN_STEM:
                w = w[: -len(suf)]
                break
        return w


STEMMER = RussianStemmer()


def stem_word(word):
    return STEMMER.stem(word)
