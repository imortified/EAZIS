# Сравнение с эталонным рефератом и повторные измерения качества и времени работы.
from collections import Counter,defaultdict
from statistics import mean,median
import re
import unicodedata
from .nlp import stem,LANGUAGES,DOMAINS
from .engine import summarize,METHODS

def metric_tokens(text,language):  # Подготовить токены ROUGE со стеммингом, сохраняя стоп-слова и числа.
    return [stem(t,language) for t in re.findall(r'[^\W_]+',unicodedata.normalize('NFC',text).lower())]

def _f1(overlap,n_candidate,n_reference):  # Рассчитать precision, recall и их гармоническое среднее F1.
    p=overlap/n_candidate if n_candidate else 0  # Доля совпадений среди токенов полученного реферата.
    r=overlap/n_reference if n_reference else 0  # Доля совпадений среди токенов эталона.
    return {'precision':p,'recall':r,'f1':2*p*r/(p+r) if p+r else 0}

def lcs_length(a,b):  # Найти длину наибольшей общей подпоследовательности для ROUGE-L.
    if len(a)>len(b):a,b=b,a
    previous=[0]*(len(a)+1)  # Хранить только предыдущую строку динамической таблицы для экономии памяти.
    for token in b:
        current=[0]
        for j,other in enumerate(a,1):
            current.append(previous[j-1]+1 if token==other else max(previous[j],current[-1]))
        previous=current
    return previous[-1]

def reference_metrics(summary,reference,language):  # Сравнить реферат с непустым эталоном по ROUGE-1 и ROUGE-L.
    if not reference.strip():return None
    a,b=metric_tokens(summary,language),metric_tokens(reference,language)
    one=sum((Counter(a)&Counter(b)).values())  # Совпадения одиночных токенов с учётом повторений.
    return {'rouge1':_f1(one,len(a),len(b)),'rougeL':_f1(lcs_length(a,b),len(a),len(b))}

def evaluate(corpus,sentence_count=10,repetitions=3):  # Проверить оба метода на коллекции и сгруппировать показатели.
    if not corpus:raise ValueError('Коллекция пуста.')
    if len(corpus)>80:raise ValueError('Для эксперимента выберите коллекцию не более 80 документов.')
    rows=[]
    for document in corpus:
        for method in METHODS:
            try:
                summarize(document,corpus,method,sentence_count) # Прогрев не учитывается в медиане.
                runs=[summarize(document,corpus,method,sentence_count) for _ in range(repetitions)]  # Повторить измерение для устойчивой оценки времени.
                result=runs[-1]
                reference=reference_metrics(result['summary'],document['reference'],document['language'])  # ROUGE считается отдельно от времени реферирования.
                rows.append({'document_id':document['id'],'title':document['title'],'language':document['language'],
                    'domain':document['domain'],'method':method,'median_ms':median(x['elapsed_ms'] for x in runs),
                    'compression':result['compression'],'redundancy':result['redundancy'],'term_coverage':result['term_coverage'],
                    'reference_kind':document['reference_kind'],'reference_metrics':reference,
                    'selected_count':result['selected_count'],'sentence_count':result['sentence_count'],
                    'rouge1_f1':reference['rouge1']['f1'] if reference else None,
                    'rougeL_f1':reference['rougeL']['f1'] if reference else None})
            except ValueError as exc:  # Ошибку одного документа записать в результат и продолжить эксперимент.
                rows.append({'document_id':document['id'],'title':document['title'],'language':document['language'],
                             'domain':document['domain'],'method':method,'error':str(exc)})
    groups=[]  # Отдельные группы: язык × предметная область × метод.
    for language in LANGUAGES:
        for domain in DOMAINS:
            for method in METHODS:
                items=[r for r in rows if r['language']==language and r['domain']==domain and r['method']==method and not r.get('error')]
                if not items:continue
                group={'language':language,'domain':domain,'method':method,'count':len(items),
                       'reference_count':sum(r['reference_metrics'] is not None for r in items)}
                for key in ['median_ms','compression','redundancy','term_coverage','rouge1_f1','rougeL_f1']:
                    values=[r[key] for r in items if r[key] is not None]
                    group[key]=mean(values) if values else None  # Усреднять только имеющиеся значения.
                groups.append(group)
    return {'rows':rows,'groups':groups,'sentence_count':sentence_count,'repetitions':repetitions,
            'timing':'Медиана повторов после одного прогрева; предобработка, веса, граф и ключевые слова, без дискового ввода и расчета ROUGE.'}
