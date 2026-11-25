import re
from dataclasses import dataclass, field

from . import db
from .document import Document, tokenize, STOPWORDS
from .stemmer import stem_word
from .spellcheck import suggest_query_fix

QUERY_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")

OPERATORS = {
    "and": "AND", "и": "AND",
    "or": "OR", "или": "OR",
    "not": "NOT", "не": "NOT",
}


@dataclass
class SearchResult:
    document_id: int
    title: str
    snippet: str            # первые 300 символов документа
    rank: float             # суммарный вес совпавших терминов
    date: str
    time: str = ""
    matched_words: list = field(default_factory=list)  # слова запроса, найденные в документе


def lex_query(query):
    tokens = []
    for raw in QUERY_TOKEN_RE.findall(query or ""):
        low = raw.lower()
        if raw == "(":
            tokens.append(("LPAREN", "("))
        elif raw == ")":
            tokens.append(("RPAREN", ")"))
        elif low in OPERATORS:
            tokens.append(("OP", OPERATORS[low]))
        else:
            tokens.append(("WORD", raw))
    return tokens


class QueryParser:
    def __init__(self, tokens, implicit="AND"):
        self.tokens = tokens
        self.pos = 0
        self.implicit = implicit

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def next(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def parse(self):
        node = self.or_expr()
        if self.peek()[0] is not None:
            raise SyntaxError("Некорректное выражение (лишние символы или скобки)")
        return node

    def or_expr(self):
        node = self.and_expr()
        while self.peek() == ("OP", "OR"):
            self.next()
            node = ("or", node, self.and_expr())
        return node

    def and_expr(self):
        node = self.not_expr()
        while True:
            kind, value = self.peek()
            if kind == "OP" and value == "AND":
                self.next()
                node = ("and", node, self.not_expr())
            elif kind in ("WORD", "LPAREN") or (kind == "OP" and value == "NOT"):
                # неявная связка между соседними терминами
                node = (self.implicit.lower(), node, self.not_expr())
            else:
                break
        return node

    def not_expr(self):
        kind, value = self.peek()
        if (kind, value) == ("OP", "NOT"):
            self.next()
            return ("not", self.not_expr())
        return self.atom()

    def atom(self):
        kind, value = self.next()
        if kind == "LPAREN":
            node = self.or_expr()
            if self.peek()[0] != "RPAREN":
                raise SyntaxError("Не закрыта скобка")
            self.next()
            return node
        if kind == "WORD":
            return ("term", value)
        raise SyntaxError("Неожиданный конец запроса или лишняя скобка")


def _term_stems(word):
    out = []
    for part in tokenize(word):
        if len(part) >= 2 and part not in STOPWORDS:
            out.append((stem_word(part), part))
    return out


class Search:
    def __init__(self, all_words_together=True, date_start=None, date_end=None,
                 search_query=""):
        self.all_words_together = all_words_together   # True -> AND, False -> OR
        self.date_start = date_start                   # 'YYYY-MM-DD' или None
        self.date_end = date_end                       # 'YYYY-MM-DD' или None
        self.search_query = search_query
        self.last_error = None                         # текст ошибки запроса
        self.unknown_terms = []                        # слова, отсутствующие в словаре
        self.suggestions = []                          # [Correction] — предлагаемые исправления опечаток
        self.suggestion_query = ""                     # запрос после исправления опечаток

    # ------------------ публичный метод ------------------ #
    def GetSearchResult(self):
        self.last_error = None
        self.unknown_terms = []
        self.suggestions = []
        self.suggestion_query = ""
        query = (self.search_query or "").strip()
        if not query:
            self.last_error = "Введите поисковый запрос."
            return []

        tokens = lex_query(query)
        if not any(k == "WORD" for k, _ in tokens):
            self.last_error = "Запрос не содержит поисковых слов."
            return []

        implicit = "AND" if self.all_words_together else "OR"
        try:
            tree = QueryParser(tokens, implicit=implicit).parse()
        except SyntaxError as exc:
            self.last_error = f"Ошибка в запросе: {exc}"
            return []

        all_docs = self._all_doc_ids()
        docs = self._eval(tree, all_docs)

        if self.unknown_terms:
            fixed, fixes = suggest_query_fix(query)
            if fixes:
                self.suggestion_query = fixed or ""
                self.suggestions = fixes

        docs = self._filter_by_date(docs)

        if not docs:
            return []

        rows = self._load_rows(docs)
        postings = self._query_postings(tokens)      # {основа: {doc: freq}}
        surface = self._surface_map(tokens)          # {основа: форма из запроса}

        results = []
        for r in rows:
            doc_id = r["document_id"]
            forms = Document.get_surface_forms(doc_id)
            matched, rank = [], 0.0
            for stem_, posts in postings.items():
                if doc_id in posts:
                    surface_form = surface.get(stem_, stem_)
                    matched.append(surface_form)
                    rank += posts[doc_id] * (Document.GetLemmInverseFrequency(stem_) or 0.0)
            if not matched:
                continue  # документ без единого слова запроса не показываем
            snippet = r["text"][:300] + ("…" if len(r["text"]) > 300 else "")
            results.append(SearchResult(
                document_id=doc_id,
                title=r["title"],
                snippet=snippet,
                rank=round(rank, 4),
                date=r["date"],
                time=r["time"],
                matched_words=matched,
            ))

        results.sort(key=lambda x: (x.rank, x.date, x.document_id), reverse=True)
        return results

    def _all_doc_ids(self):
        conn = db.get_connection()
        try:
            return {r["document_id"] for r in
                    conn.execute("SELECT document_id FROM documents").fetchall()}
        finally:
            conn.close()

    def _eval(self, node, all_docs):
        op = node[0]
        if op == "term":
            stems = _term_stems(node[1])
            if not stems:
                return set(all_docs)          # стоп-слово не влияет на отбор
            result = None
            for stem_, _part in stems:        # дефисные части объединяются AND
                posts = Document.get_postings(stem_)
                if not posts:
                    self.unknown_terms.append(node[1])
                    return set()              # отказ: термина нет в индексе
                sset = set(posts.keys())
                result = sset if result is None else (result & sset)
            return result or set()
        if op == "not":
            return all_docs - self._eval(node[1], all_docs)
        if op == "and":
            return self._eval(node[1], all_docs) & self._eval(node[2], all_docs)
        if op == "or":
            return self._eval(node[1], all_docs) | self._eval(node[2], all_docs)
        raise ValueError(f"Неизвестный узел: {op}")

    def _filter_by_date(self, docs):
        if not self.date_start and not self.date_end:
            return docs
        conn = db.get_connection()
        try:
            sql = "SELECT document_id FROM documents WHERE 1=1"
            params = []
            if self.date_start:
                sql += " AND date >= ?"
                params.append(self.date_start)
            if self.date_end:
                sql += " AND date <= ?"
                params.append(self.date_end)
            allowed = {r["document_id"] for r in conn.execute(sql, params).fetchall()}
            return docs & allowed
        finally:
            conn.close()

    def _load_rows(self, docs):
        conn = db.get_connection()
        try:
            marks = ",".join("?" * len(docs))
            return conn.execute(
                f"SELECT * FROM documents WHERE document_id IN ({marks})",
                tuple(docs),
            ).fetchall()
        finally:
            conn.close()

    def _query_postings(self, tokens):
        postings = {}
        for kind, value in tokens:
            if kind != "WORD":
                continue
            for stem_, _part in _term_stems(value):
                if stem_ not in postings:
                    posts = Document.get_postings(stem_)
                    if posts:
                        postings[stem_] = posts
        return postings

    def _surface_map(self, tokens):
        surface = {}
        for kind, value in tokens:
            if kind != "WORD":
                continue
            for stem_, part in _term_stems(value):
                surface.setdefault(stem_, part)
        return surface
