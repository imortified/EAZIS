# Хранение документов, рефератов и результатов эксперимента в локальной базе SQLite.
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime,timezone
import sqlite3
import hashlib
import json

# SQL-схема таблиц документов, рефератов, экспериментов и служебных меток.
SCHEMA='''
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
 id INTEGER PRIMARY KEY,title TEXT NOT NULL,text TEXT NOT NULL,language TEXT NOT NULL CHECK(language IN ('en','fr')),
 domain TEXT NOT NULL CHECK(domain IN ('medicine','art')),source_url TEXT NOT NULL DEFAULT '',author TEXT NOT NULL DEFAULT '',
 license TEXT NOT NULL DEFAULT '',reference TEXT NOT NULL DEFAULT '',reference_kind TEXT NOT NULL DEFAULT '',
 original_name TEXT NOT NULL,original BLOB NOT NULL,mime TEXT NOT NULL,content_hash TEXT NOT NULL,created TEXT NOT NULL,
 UNIQUE(content_hash,language));
CREATE TABLE IF NOT EXISTS summaries(
 id INTEGER PRIMARY KEY,document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 method TEXT NOT NULL,sentence_count INTEGER NOT NULL,corpus_signature TEXT NOT NULL,result_json TEXT NOT NULL,created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evaluations(id INTEGER PRIMARY KEY,corpus_signature TEXT NOT NULL,result_json TEXT NOT NULL,created TEXT NOT NULL);
'''

def now():  # Получить время UTC в строковом формате ISO с точностью до секунды.
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

class Repository:  # Объединить операции с базой и преобразование записей в словари Python.
    def __init__(self,path):  # Подготовить каталог базы и создать отсутствующие таблицы.
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as c:c.executescript(SCHEMA)

    @contextmanager
    def connect(self):  # Открыть соединение, завершить транзакцию и обязательно закрыть базу.
        c=sqlite3.connect(self.path,timeout=15);c.row_factory=sqlite3.Row
        c.execute('PRAGMA foreign_keys=ON')  # Включить внешние ключи и каскадное удаление рефератов.
        try:
            yield c;c.commit()  # После успешного блока with сохранить изменения.
        except Exception:
            c.rollback();raise  # При ошибке отменить транзакцию и передать исключение вызывающему коду.
        finally:c.close()

    def documents(self,language=None):  # Прочитать коллекцию без исходных байтов, при необходимости отфильтровав язык.
        with self.connect() as c:
            sql='SELECT id,title,text,language,domain,source_url,author,license,reference,reference_kind,original_name,content_hash,created FROM documents'
            return [dict(r) for r in c.execute(sql+(' WHERE language=?' if language else '')+' ORDER BY id',(language,) if language else ())]

    def document(self,document_id):  # Прочитать полную запись документа, включая оригинал файла в BLOB.
        with self.connect() as c:
            row=c.execute('SELECT * FROM documents WHERE id=?',(document_id,)).fetchone()
            return dict(row) if row else None

    def add_document(self,*,title,text,language,domain,source_url='',author='',license='',reference='',reference_kind='',original_name='document.txt',original=None,mime='text/plain'):  # Добавить документ либо вернуть существующий дубль.
        digest=hashlib.sha256(text.encode('utf-8')).hexdigest()  # Хеш текста вместе с языком определяет дубликат.
        with self.connect() as c:
            found=c.execute('SELECT id FROM documents WHERE content_hash=? AND language=?',(digest,language)).fetchone()
            if found:return found['id'],False
            # Значения передаются параметрами SQL; при ручном вводе оригиналом служат байты текста.
            cur=c.execute('INSERT INTO documents(title,text,language,domain,source_url,author,license,reference,reference_kind,original_name,original,mime,content_hash,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (title,text,language,domain,source_url,author,license,reference,reference_kind,original_name,original if original is not None else text.encode('utf-8'),mime,digest,now()))
            return cur.lastrowid,True

    def delete(self,document_id):  # Удалить документ; внешний ключ удалит связанные с ним рефераты.
        with self.connect() as c:c.execute('DELETE FROM documents WHERE id=?',(document_id,))

    def update_reference(self,document_id,reference):  # Обновить эталон и отметить его как введённый пользователем.
        with self.connect() as c:
            c.execute("UPDATE documents SET reference=?,reference_kind='user_reference' WHERE id=?",(reference,document_id))

    def signature(self):  # Рассчитать подпись текущей коллекции для проверки актуальности результатов.
        # Изменение эталонного реферата также делает сохранённые оценки устаревшими.
        payload=[(d['id'],d['content_hash'],d['language'],d['domain'],d['reference']) for d in self.documents()]
        return hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode()).hexdigest()

    def save_summary(self,document_id,result):  # Сохранить результат реферирования в JSON вместе с подписью коллекции.
        with self.connect() as c:
            cur=c.execute('INSERT INTO summaries(document_id,method,sentence_count,corpus_signature,result_json,created) VALUES(?,?,?,?,?,?)',
                (document_id,result['method'],result['requested_count'],self.signature(),json.dumps(result,ensure_ascii=False),now()))
            return cur.lastrowid

    def summary(self,summary_id):  # Прочитать реферат, восстановить JSON и определить, устарел ли результат.
        with self.connect() as c:
            row=c.execute('SELECT * FROM summaries WHERE id=?',(summary_id,)).fetchone()
            if not row:return None
            value=dict(row);value['result']=json.loads(value.pop('result_json'))
            value['stale']=value['corpus_signature']!=self.signature()  # Разные подписи означают изменение коллекции или эталонов.
            return value

    def save_evaluation(self,result):  # Сохранить полный результат эксперимента и состояние коллекции.
        with self.connect() as c:
            c.execute('INSERT INTO evaluations(corpus_signature,result_json,created) VALUES(?,?,?)',
                      (self.signature(),json.dumps(result,ensure_ascii=False),now()))

    def evaluation(self):  # Вернуть последний эксперимент с признаком актуальности.
        with self.connect() as c:
            row=c.execute('SELECT * FROM evaluations ORDER BY id DESC LIMIT 1').fetchone()
            if not row:return None
            value=dict(row);value['result']=json.loads(value.pop('result_json'))
            value['stale']=value['corpus_signature']!=self.signature()
            return value

    def metadata(self,key,value=None):  # Прочитать служебную метку либо записать её при переданном значении.
        with self.connect() as c:
            if value is not None:
                c.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)',(key,value))
                return value
            row=c.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
            return row['value'] if row else None
