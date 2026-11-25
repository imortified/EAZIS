# Чтение поддерживаемых форматов и проверка текста до сохранения в базе.
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
import re
from bs4 import BeautifulSoup
from .nlp import normalize_input

MAX_BYTES=8*1024*1024  # Предел размера одного исходного файла — 8 МиБ.
MAX_CHARS=120000  # Предел длины извлечённого текста в символах.
EXTENSIONS={'.txt':'text/plain','.md':'text/markdown','.html':'text/html','.htm':'text/html',
            '.pdf':'application/pdf','.docx':'application/vnd.openxmlformats-officedocument.wordprocessingml.document'}

def decode(raw):  # Преобразовать байты текста в строку, проверяя UTF-16, UTF-8 и cp1252.
    if raw.startswith((b'\xff\xfe',b'\xfe\xff')):return raw.decode('utf-16')  # Маркер в начале файла определяет UTF-16.
    try:return raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        try:return raw.decode('cp1252')
        except UnicodeDecodeError:raise ValueError('Не удалось прочитать кодировку. Сохраните текст в UTF-8.')

def source_url(value):  # Разрешить пустую ссылку либо полный адрес HTTP/HTTPS с именем сервера.
    value=(value or '').strip()
    if value and (urlsplit(value).scheme not in {'http','https'} or not urlsplit(value).netloc):
        raise ValueError('Источник должен быть ссылкой http:// или https://.')
    return value

def validate_text(text):  # Нормализовать текст и проверить длину и признаки повреждённой кодировки.
    text=normalize_input(text)
    if len(text)<30:raise ValueError('Документ пуст или слишком короткий: минимум 30 символов.')
    if len(text)>MAX_CHARS:raise ValueError(f'Максимум {MAX_CHARS} символов. Загрузите отдельный раздел.')
    if text.count('\ufffd')>max(2,len(text)//200):raise ValueError('Текст содержит слишком много поврежденных символов.')  # U+FFFD обозначает ошибку декодирования.
    return text

def read_upload(filename,raw):  # Извлечь текст по расширению файла и вернуть его вместе с MIME-типом.
    if len(raw)>MAX_BYTES:raise ValueError('Файл больше 8 МБ.')
    extension=Path(filename).suffix.lower()
    if extension not in EXTENSIONS:raise ValueError('Поддерживаются TXT, MD, HTML, PDF и DOCX.')
    try:
        if extension=='.pdf':  # Читается готовый текстовый слой; распознавание сканов не выполняется.
            from pypdf import PdfReader
            reader=PdfReader(BytesIO(raw))
            if reader.is_encrypted:raise ValueError('Зашифрованные PDF не поддерживаются.')
            text='\n\n'.join(page.extract_text() or '' for page in reader.pages)
            if not text.strip():raise ValueError('В PDF нет текстового слоя. OCR не выполняется.')
        elif extension=='.docx':  # Из DOCX извлекаются непустые абзацы основного текста.
            from docx import Document
            document=Document(BytesIO(raw))
            text='\n\n'.join(p.text for p in document.paragraphs if p.text.strip())
        elif extension in {'.html','.htm'}:
            soup=BeautifulSoup(raw,'html.parser')
            for tag in soup.select('script,style,nav,header,footer,aside'):tag.decompose()  # Удалить оформление, код и навигацию.
            main=soup.find('article') or soup.find('main') or soup.body or soup  # Предпочесть основное содержимое страницы.
            paragraphs=main.find_all(['p','h1','h2','h3'])
            text='\n\n'.join(p.get_text(' ',strip=True) for p in paragraphs) if paragraphs else main.get_text(' ',strip=True)
        else:text=decode(raw)  # TXT и MD читаются как обычный текст.
    except ValueError:raise  # Сохранить понятные сообщения проверок, сформированные выше.
    except Exception as exc:raise ValueError('Не удалось прочитать файл. Проверьте его формат и целостность.') from exc
    return validate_text(text),EXTENSIONS[extension]
