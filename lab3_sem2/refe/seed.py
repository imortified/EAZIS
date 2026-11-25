# Первоначальная загрузка комплектных документов из манифеста и локальных текстовых файлов.
from pathlib import Path
import json

def import_corpus(repository,root):  # Добавить документы из манифеста, пропуская уже существующие тексты.
    manifest=Path(root)/'data'/'corpus'/'manifest.json'
    if not manifest.exists():return 0
    items=json.loads(manifest.read_text(encoding='utf-8'))['documents']  # В манифесте хранятся пути, языки, области и сведения об источниках.
    added=0
    for item in items:
        text=(manifest.parent/item['text_file']).read_text(encoding='utf-8')
        _,created=repository.add_document(title=item['title'],text=text,language=item['language'],domain=item['domain'],
            source_url=item['source_url'],author=item['author'],license=item['license'],reference=item.get('reference',''),
            reference_kind=item.get('reference_kind',''),original_name=Path(item['text_file']).name)
        added+=created  # True увеличивает число новых документов на единицу, False — на ноль.
    repository.metadata('bundled_corpus_imported','1')  # Отметить выполнение автоматического импорта.
    return added
