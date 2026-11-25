# Веб-приложение: загрузка документов, реферирование, оценка и экспорт результатов.
from pathlib import Path
from io import BytesIO,StringIO
from math import cos,sin,pi
from datetime import datetime,timezone
import argparse
import csv
import os
import secrets
from flask import Flask,render_template,request,redirect,url_for,flash,abort,send_file
from werkzeug.exceptions import RequestEntityTooLarge
from refe.storage import Repository
from refe.nlp import LANGUAGES,DOMAINS,split_sentences
from refe.importing import read_upload,validate_text,source_url
from refe.engine import summarize,METHODS
from refe.evaluation import evaluate,reference_metrics
from refe.seed import import_corpus

ROOT=Path(__file__).resolve().parent  # Пути к данным считаются от папки приложения.

def create_app(test_config=None):  # Создать Flask-приложение, настроить базу и зарегистрировать маршруты.
    app=Flask(__name__)
    # Секрет используется для сообщений сессии; общий размер загрузки ограничен 20 МиБ.
    app.config.update(SECRET_KEY=os.environ.get('LAB3_SECRET',secrets.token_hex(24)),
                      DATABASE=str(ROOT/'data'/'lab3.db'),MAX_CONTENT_LENGTH=20*1024*1024,AUTO_SEED=True)
    if test_config:app.config.update(test_config)  # Тесты могут подставить отдельную временную базу.
    repo=Repository(app.config['DATABASE']);app.extensions['repository']=repo
    # Комплектная коллекция добавляется автоматически один раз, пока нет служебной отметки.
    if app.config['AUTO_SEED'] and not repo.metadata('bundled_corpus_imported'):import_corpus(repo,ROOT)

    @app.context_processor
    def shared():  # Передать во все шаблоны подписи языков, областей, методов и типов эталона.
        return {'languages':LANGUAGES,'domains':DOMAINS,'methods':METHODS,
        'reference_kinds':{'author_abstract':'Авторская аннотация','source_lead_proxy':'Вступление источника — условный ориентир','user_reference':'Ручной эталон'}}

    def document_or_404(document_id):  # Получить документ по идентификатору или завершить запрос ошибкой 404.
        document=repo.document(document_id)
        if not document:abort(404)
        return document

    def parameters():  # Прочитать метод и число предложений из формы и проверить допустимые значения.
        method=request.form.get('method','textrank')
        count=int(request.form.get('sentence_count','10'))
        if method not in METHODS:raise ValueError('Неизвестный метод.')
        if not 1<=count<=50:raise ValueError('Укажите от 1 до 50 предложений.')
        return method,count

    def metadata():  # Проверить выбранные пользователем язык, область и ссылку на источник.
        language=request.form.get('language','en');domain=request.form.get('domain','medicine')
        if language not in LANGUAGES or domain not in DOMAINS:raise ValueError('Выберите язык и предметную область.')
        return {'language':language,'domain':domain,'source_url':source_url(request.form.get('source_url',''))}

    # Декораторы app.get и app.post связывают адрес страницы с функцией обработки запроса.
    @app.get('/')
    def index():  # Показать главную страницу и количество документов в каждой группе.
        documents=repo.documents()
        return render_template('index.html',documents=documents,
            groups=[{'language':lang,'domain':domain,'count':sum(d['language']==lang and d['domain']==domain for d in documents)} for lang in LANGUAGES for domain in DOMAINS])

    @app.get('/documents')
    def documents_page():  # Показать коллекцию и формы добавления документов.
        return render_template('documents.html',documents=repo.documents())

    @app.post('/documents/add')
    def document_add():  # Сохранить текст, вставленный пользователем в форму.
        try:
            meta=metadata();text=validate_text(request.form.get('text',''))
            title=request.form.get('title','').strip()[:250]
            if not title:raise ValueError('Введите заголовок.')
            doc_id,added=repo.add_document(title=title,text=text,**meta)  # Повторный текст получает идентификатор существующей записи.
            flash('Документ добавлен.' if added else 'Такой текст уже есть в коллекции.','ok' if added else 'warn')
            return redirect(url_for('document_view',document_id=doc_id))
        except ValueError as exc:flash(str(exc),'error')  # Ошибку ввода показать в интерфейсе.
        return redirect(url_for('documents_page'))

    @app.post('/documents/upload')
    def document_upload():  # Прочитать загруженные файлы и сохранить текст вместе с исходными байтами.
        try:
            meta=metadata();files=[f for f in request.files.getlist('files') if f.filename]
            if not files:raise ValueError('Выберите файл.')
            if len(files)>20:raise ValueError('За одну загрузку допускается до 20 файлов.')
            added=0  # Счётчик только новых документов, без дубликатов.
            for file in files:
                try:
                    raw=file.read();text,mime=read_upload(file.filename,raw)  # Чтение формата вынесено в refe/importing.py.
                    _,created=repo.add_document(title=Path(file.filename).stem[:250],text=text,**meta,
                        original_name=Path(file.filename).name,original=raw,mime=mime)
                    added+=created
                except ValueError as exc:flash(f'{file.filename}: {exc}','error')  # Сбой одного файла не останавливает остальные.
            flash(f'Добавлено документов: {added}. Повторяющиеся тексты пропущены.','ok')
        except ValueError as exc:flash(str(exc),'error')
        return redirect(url_for('documents_page'))

    @app.post('/corpus/import')
    def corpus_import():  # Добавить отсутствующие документы из комплектного манифеста.
        flash(f'Добавлено документов из комплекта: {import_corpus(repo,ROOT)}.','ok')
        return redirect(url_for('documents_page'))

    @app.get('/documents/<int:document_id>')
    def document_view(document_id):  # Показать исходный текст, его предложения и настройки реферирования.
        document=document_or_404(document_id)
        return render_template('document.html',document=document,sentences=split_sentences(document['text']))

    @app.get('/documents/<int:document_id>/original')
    def document_original(document_id):  # Отдать пользователю сохранённый оригинал файла.
        document=document_or_404(document_id)
        return send_file(BytesIO(document['original']),mimetype=document['mime'],as_attachment=True,
                         download_name=document['original_name'].replace('\\','_'))

    @app.post('/documents/<int:document_id>/delete')
    def document_delete(document_id):  # Удалить документ; связанные рефераты удаляются каскадно в SQLite.
        document_or_404(document_id);repo.delete(document_id);flash('Документ удален.','ok')
        return redirect(url_for('documents_page'))

    @app.post('/documents/<int:document_id>/reference')
    def reference_update(document_id):  # Сохранить ручной эталонный реферат для последующей оценки ROUGE.
        document_or_404(document_id)
        reference=request.form.get('reference','').strip()
        if len(reference)>12000:flash('Эталон должен быть не длиннее 12000 символов.','error')
        else:repo.update_reference(document_id,reference);flash('Эталон сохранен. Оценки можно пересчитать.','ok')
        return redirect(url_for('document_view',document_id=document_id))

    @app.post('/documents/<int:document_id>/summarize')
    def summarize_document(document_id):  # Построить реферат выбранным методом и сохранить снимок результата.
        document=document_or_404(document_id)
        try:
            # Весь корпус нужен базовому методу для расчёта документной частоты и IDF.
            method,count=parameters();result=summarize(document,repo.documents(),method,count)
            summary_id=repo.save_summary(document_id,result)
            return redirect(url_for('summary_view',summary_id=summary_id))
        except ValueError as exc:flash(str(exc),'error')
        return redirect(url_for('document_view',document_id=document_id))

    @app.post('/summarize')
    def summarize_selected():  # Получить документ из списка на главной странице и запустить реферирование.
        try:document_id=int(request.form.get('document_id','0'))
        except ValueError:abort(404)
        return summarize_document(document_id)

    @app.get('/summaries/<int:summary_id>')
    def summary_view(summary_id):  # Показать реферат, ключевые фразы, граф и сравнение с эталоном.
        saved=repo.summary(summary_id)
        if not saved:abort(404)
        document=document_or_404(saved['document_id']);result=saved['result']
        nodes=[]
        # Координаты размещают отображаемые вершины по окружности; ранги графа уже рассчитаны.
        for k,index in enumerate(result['graph_nodes']):
            angle=2*pi*k/max(1,len(result['graph_nodes']))-pi/2
            nodes.append({'index':index,'x':300+225*cos(angle),'y':285+220*sin(angle),
                          'score':result['sentences'][index]['score'],'text':result['sentences'][index]['text'],
                          'selected':result['sentences'][index]['selected']})
        return render_template('summary.html',saved=saved,result=result,document=document,graph_nodes=nodes,
            graph_positions={n['index']:n for n in nodes},reference=reference_metrics(result['summary'],document['reference'],document['language']))

    @app.route('/compare',methods=['GET','POST'])
    def compare():  # Построить два реферата одного документа при одинаковом числе предложений.
        documents=repo.documents();selected=None;results=[];error=None;count=10
        if request.method=='POST':
            try:
                selected=document_or_404(int(request.form.get('document_id','0')))
                _,count=parameters()
                results=[summarize(selected,documents,method,count) for method in METHODS]
            except ValueError as exc:error=str(exc)
        return render_template('compare.html',documents=documents,document=selected,results=results,error=error,count=count)

    @app.get('/metrics')
    def metrics():  # Показать последний сохранённый эксперимент и статистику коллекции.
        return render_template('metrics.html',snapshot=repo.evaluation(),documents=repo.documents())

    @app.post('/metrics/run')
    def metrics_run():  # Проверить оба метода на всей коллекции и сохранить результаты эксперимента.
        try:
            _,count=parameters();data=evaluate(repo.documents(),count,3)  # Три измерения после прогрева.
            repo.save_evaluation(data);flash('Эксперимент завершен: оба метода проверены на одной коллекции.','ok')
        except ValueError as exc:flash(str(exc),'error')
        return redirect(url_for('metrics'))

    @app.get('/metrics/export.csv')
    def metrics_export():  # Выгрузить построчные результаты последнего эксперимента в CSV.
        saved=repo.evaluation()
        if not saved:abort(404)
        stream=StringIO();fields=['document_id','title','language','domain','method','median_ms','compression','redundancy','term_coverage','rouge1_f1','rougeL_f1','reference_kind','error']
        # Список fields задаёт порядок колонок; вложенные поля результата в CSV не попадают.
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(saved['result']['rows'])
        return send_file(BytesIO(stream.getvalue().encode('utf-8-sig')),mimetype='text/csv',as_attachment=True,download_name='lab3_metrics.csv')

    @app.get('/summaries/<int:summary_id>/export.<extension>')
    def summary_export(summary_id,extension):  # Собрать и скачать сохранённый реферат в TXT или DOCX.
        saved=repo.summary(summary_id)
        if not saved or extension not in {'txt','docx'}:abort(404)
        d=document_or_404(saved['document_id']);r=saved['result']
        title=d['title'];keywords=', '.join(k['text'] for k in r['keywords'])
        source=d['source_url'] or 'Локальный документ'
        text=f'{title}\n\nМетод: {r["method_label"]}\nИсточник: {source}\n\nКлассический реферат\n\n{r["summary"]}\n\nКлючевые слова\n{keywords}\n'
        if extension=='txt':  # UTF-8 с маркером кодировки упрощает открытие русского текста в редакторах.
            return send_file(BytesIO(text.encode('utf-8-sig')),mimetype='text/plain',as_attachment=True,download_name=f'summary_{summary_id}.txt')
        # DOCX создаётся в памяти: отдельный временный файл на диске не требуется.
        from docx import Document
        from docx.shared import Pt
        out=Document();out.styles['Normal'].font.name='Times New Roman';out.styles['Normal'].font.size=Pt(12)
        out.add_heading(title,0);out.add_paragraph(f'Метод: {r["method_label"]}\nИсточник: {source}')
        out.add_heading('Классический реферат',1)
        for paragraph in r['summary'].split('\n\n'):out.add_paragraph(paragraph)
        out.add_heading('Ключевые слова',1);out.add_paragraph(keywords)
        buffer=BytesIO();out.save(buffer);buffer.seek(0)
        return send_file(buffer,mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',as_attachment=True,download_name=f'summary_{summary_id}.docx')

    @app.get('/help')
    def help_page():  # Показать справку об алгоритмах и работе с приложением.
        return render_template('help.html')

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):  # Обработать превышение общего размера HTTP-загрузки.
        return render_template('error.html',message='Общий размер загрузки больше 20 МБ.'),413

    @app.errorhandler(404)
    def not_found(_):  # Показать понятное сообщение для отсутствующей страницы или записи.
        return render_template('error.html',message='Документ или результат не найден.'),404

    return app

if __name__=='__main__':  # При прямом запуске прочитать порт и запустить локальный сервер.
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=5003)
    args=parser.parse_args();create_app().run(host='127.0.0.1',port=args.port,debug=False)
