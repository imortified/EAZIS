import re
import PyPDF2
import pymorphy3
from sentence_transformers import SentenceTransformer

print("[INFO] Загрузка модели эмбеддингов...")
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print("[INFO] Модель эмбеддингов загружена.")

morph = pymorphy3.MorphAnalyzer()
print("[INFO] pymorphy3 инициализирован.")


def lemmatize_text(text): # приводим слова к начальной форме для улучшения семантического поиска
    words = re.findall(r'\b\w+\b', text.lower())
    lemmas = []
    for word in words:
        parsed = morph.parse(word)
        if parsed:
            lemmas.append(parsed[0].normal_form)
        else:
            lemmas.append(word)
    return ' '.join(lemmas)


def extract_keywords(text, pos_filter=None): # Извлечение ключевых слов с фильтрацией по части речи
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = []
    for word in words:
        parsed = morph.parse(word)
        if parsed:
            p = parsed[0]
            if pos_filter is None or p.tag.POS in pos_filter:
                keywords.append(p.normal_form)
    return keywords


def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_text(text, chunk_size=512, overlap=128):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sent_len = len(sentence)
        if current_length + sent_len > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            prev_text = ' '.join(current_chunk)
            overlap_text = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
            current_chunk = [overlap_text, sentence]
            current_length = len(overlap_text) + sent_len
        else:
            current_chunk.append(sentence)
            current_length += sent_len + 1
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def get_embedding(text):
    return embedder.encode(text, convert_to_numpy=True)
