# Подготовка EN/FR-текста: абзацы, предложения, токены, стоп-слова и стеммы.
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import unicodedata
from nltk.stem.snowball import SnowballStemmer

LANGUAGES = {'en': 'Английский', 'fr': 'Французский'}
DOMAINS = {'medicine': 'Медицина', 'art': 'Художественная критика'}
TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)  # Буквенные слова с возможным апострофом внутри.
ABBREVIATIONS = {'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'st.', 'vs.', 'etc.',
                 'fig.', 'no.', 'e.g.', 'i.e.', 'al.', 'm.', 'mme.', 'mlle.',
                 'pr.', 'p.', 'pp.', 'vol.', 'cf.', 'env.', 'av.', 'j.-c.'}
STOP_PATH = Path(__file__).parent / 'stopwords'
STEMMERS = {'en': SnowballStemmer('english'), 'fr': SnowballStemmer('french')}

@dataclass(frozen=True)
class Sentence:  # Предложение вместе с точными символьными границами и номером абзаца.
    index: int
    text: str
    start: int
    end: int  # Правая граница не включается, как при срезе text[start:end].
    paragraph: int
    paragraph_start: int
    paragraph_end: int

    def positions(self, document_length):  # Вернуть позиционные множители для документа и абзаца.
        return (1 - self.start / max(1, document_length),
                1 - (self.start - self.paragraph_start) /
                max(1, self.paragraph_end - self.paragraph_start))

@lru_cache(maxsize=2)
def stopwords(language):  # Прочитать локальный стоп-лист; lru_cache сохраняет его в памяти.
    if language not in LANGUAGES: raise ValueError('Поддерживаются только EN и FR.')
    return frozenset((STOP_PATH / f'{language}.txt').read_text(encoding='utf-8').split())

@lru_cache(maxsize=30000)
def stem(word, language):  # Получить основу слова алгоритмом Snowball выбранного языка.
    return STEMMERS[language].stem(word)

def words(text, language, filtered=True):  # Вернуть пары «словоформа, стемм» с необязательным удалением стоп-слов.
    if language not in LANGUAGES: raise ValueError('Неизвестный язык.')
    excluded = stopwords(language) if filtered else frozenset()
    output = []
    for match in TOKEN.finditer(unicodedata.normalize('NFC', text).lower()):  # NFC сохраняет единое представление букв с диакритикой.
        word = match.group()
        if language == 'fr': word = re.sub(r"^(?:l|d|j|t|m|s|n|c|qu)['’]", '', word)  # Удалить сокращённую служебную часть перед апострофом, например l' или d'.
        if len(word) < 2 or word in excluded: continue
        output.append((word, stem(word, language)))
    return output

def counts(text, language):  # Подсчитать частоты значимых стеммов с помощью Counter.
    return Counter(st for _, st in words(text, language))

def _is_end(text, punctuation_start, punctuation_end, paragraph_end):  # Проверить, завершает ли знак пунктуации предложение.
    punctuation = text[punctuation_start:punctuation_end]
    if '.' not in punctuation or len(punctuation) > 1: return True
    if punctuation_start and punctuation_end < paragraph_end:
        if text[punctuation_start-1].isdigit() and text[punctuation_end].isdigit(): return False  # Точка между цифрами не завершает предложение.
    left = text[:punctuation_end]
    token = re.search(r"[\w.\-]+\.$", left)
    token = token.group().lower() if token else ''
    if token in ABBREVIATIONS: return False  # Известные сокращения не считаются концом предложения.
    if re.fullmatch(r'[a-zà-öø-ÿ]\.', token): return False  # Не разделять текст после инициала.
    if re.fullmatch(r'(?:[a-z]\.){2,}', token): return False
    return True

def split_sentences(text):  # Выделить исходные предложения и их границы; пустые строки разделяют абзацы.
    sentences = []
    paragraphs = list(re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)', text))  # Найти абзацы без пересборки исходной строки.
    for p_index, paragraph in enumerate(paragraphs):
        p_start, p_end = paragraph.span()
        while p_end > p_start and text[p_end-1].isspace(): p_end -= 1
        start = p_start
        boundaries = []
        for punct in re.finditer(r'[.!?]+', text[p_start:p_end]):
            a,b = p_start+punct.start(), p_start+punct.end()
            if not _is_end(text,a,b,p_end): continue
            end = b
            while end < p_end and text[end] in '\"\'»”’)]': end += 1  # Закрывающие кавычки и скобки входят в предложение.
            if end < p_end and not text[end].isspace(): continue
            # Не разрывать предложение после сокращения, если далее идёт строчная буква.
            following = text[end:p_end].lstrip()
            if following and following[0].islower() and text[a:b] == '.': continue
            boundaries.append(end)
        if not boundaries or boundaries[-1] != p_end: boundaries.append(p_end)  # Сохранить хвост абзаца даже без конечной точки.
        for end in boundaries:
            while start < end and text[start].isspace(): start += 1
            final = end
            while final > start and text[final-1].isspace(): final -= 1
            if final > start:
                sentences.append(Sentence(len(sentences),text[start:final],start,final,p_index,p_start,p_end))
            start = end
    return sentences

def normalize_input(text):  # Нормализовать Unicode и переводы строк, удалить нулевые символы и внешние пробелы.
    return unicodedata.normalize('NFC', text.replace('\r\n','\n').replace('\r','\n').replace('\x00','')).strip()
