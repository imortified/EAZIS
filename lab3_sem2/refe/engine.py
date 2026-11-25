# Алгоритмы Sentence Extraction и TextRank, отбор предложений и ключевых фраз.
from collections import Counter, defaultdict
from itertools import combinations
from math import log, sqrt
from statistics import mean
from time import perf_counter
from .nlp import split_sentences, counts, words

METHODS = {'baseline': 'Sentence Extraction', 'textrank': 'TextRank'}
MAX_SENTENCES = 450  # Ограничение сдерживает затраты на попарное сравнение предложений.

def corpus_weights(documents, language):  # Рассчитать IDF стеммов по документам выбранного языка.
    counters = [counts(d['text'], language) for d in documents if d['language'] == language]
    df = Counter(term for counter in counters for term in counter)  # Стемм учитывается один раз на документ.
    n = len(counters)
    return n, {term: log(n/frequency) for term,frequency in df.items()} if n else {}

def cosine(a,b):  # Вычислить косинусное сходство двух разреженных векторов признаков.
    if len(a) > len(b): a,b = b,a  # Перебирать меньший словарь выгоднее по времени.
    denominator = sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values()))
    return sum(value*b.get(key,0) for key,value in a.items()) / denominator if denominator else 0.0  # Нулевой вектор не создаёт сходства.

def pagerank(adjacency, damping=.85, tolerance=1e-9, max_iterations=250):  # Итерационно распределить значимость по взвешенному графу.
    n = len(adjacency)
    if not n: return [], 0
    scores = [1/n]*n  # Вначале все вершины равнозначны, сумма рангов равна единице.
    sums = [sum(row.values()) for row in adjacency]  # Общий вес исходящих рёбер каждой вершины.
    for iteration in range(1,max_iterations+1):
        dangling = sum(scores[i] for i,total in enumerate(sums) if not total)  # Масса вершин без исходящих рёбер.
        # Базовый переход и масса изолированных вершин распределяются между всеми вершинами.
        new = [(1-damping)/n+damping*dangling/n]*n
        for i,row in enumerate(adjacency):
            if sums[i]:
                factor = damping*scores[i]/sums[i]  # Передать ранг соседям пропорционально весам рёбер.
                for j,weight in row.items(): new[j] += factor*weight
        delta = sum(abs(a-b) for a,b in zip(new,scores))  # Суммарное изменение рангов за итерацию.
        scores = new
        if delta < tolerance: return scores,iteration
    return scores,max_iterations

def sentence_vectors(sentence_counts):  # Построить TF-IDF-векторы предложений с добавлением единицы при сглаживании.
    n = len(sentence_counts)
    df = Counter(t for c in sentence_counts for t in c)  # Здесь DF считается по предложениям одного текста.
    return [{t:frequency*(log((1+n)/(1+df[t]))+1) for t,frequency in c.items()} for c in sentence_counts]

def sentence_graph(vectors):  # Связать пары предложений рёбрами с положительным косинусным сходством.
    adjacency = [{} for _ in vectors]
    for i,j in combinations(range(len(vectors)),2):  # Каждая неупорядоченная пара рассматривается один раз.
        similarity = cosine(vectors[i],vectors[j])
        if similarity > 1e-12:
            adjacency[i][j]=similarity; adjacency[j][i]=similarity  # Граф неориентированный, связь записывается в обе стороны.
    return adjacency

def keyword_scores(text, language, method, document_counts, term_weights):  # Оценить слова и выбрать до 12 ключевых слов и фраз.
    if method == 'baseline':
        scores = term_weights
        # Формула оценки предложений остаётся неизменной; переход к частотам
        # при полностью нулевых IDF применяется только для ключевых слов.
        if not any(scores.values()): scores = dict(document_counts)
    else:
        stream = [st for _,st in words(text,language)]
        terms = list(dict.fromkeys(stream))  # Уникальные стеммы с сохранением порядка первого появления.
        if len(terms)>5000: terms=terms[:5000]
        indexes = {t:i for i,t in enumerate(terms)}
        graph = [defaultdict(float) for _ in terms]
        for i,t in enumerate(stream):
            if t not in indexes: continue
            for other in stream[i+1:i+5]:  # Связать слово с четырьмя следующими значимыми словами.
                if t != other and other in indexes:
                    a,b=indexes[t],indexes[other]
                    graph[a][b]+=1;graph[b][a]+=1
        ranks,_=pagerank(graph)  # Это отдельный граф слов, а не граф предложений.
        scores=dict(zip(terms,ranks))
    candidates = {}
    for sentence in split_sentences(text):
        stream = words(sentence.text,language)
        # Слова фразы должны идти подряд в исходном тексте, без пропуска стоп-слов.
        raw = words(sentence.text,language,filtered=False)
        allowed = {surface for surface,_ in stream}
        for length in (1,2,3):  # Рассмотреть одиночные слова и сочетания из двух или трёх слов.
            for i in range(len(raw)-length+1):
                seq=raw[i:i+length]
                if not all(surface in allowed for surface,_ in seq): continue
                key=tuple(st for _,st in seq)  # Разные словоформы объединяются по последовательности стеммов.
                phrase=' '.join(surface for surface,_ in seq)
                item=candidates.setdefault(key,{'text':phrase,'frequency':0,'stems':key})
                item['frequency']+=1
                if len(phrase)<len(item['text']): item['text']=phrase
    ranked=[]
    for key,item in candidates.items():
        if len(key)>1 and item['frequency']<2: continue  # Многословная фраза должна встретиться минимум дважды.
        score=sum(scores.get(t,0) for t in key)/sqrt(len(key))
        ranked.append({**item,'score':score})
    ranked.sort(key=lambda x:(-x['score'],-x['frequency'],x['text']))
    chosen=[]
    for item in ranked:
        current=set(item['stems'])
        # Коэффициент Жаккара отсекает фразы, слишком похожие на уже выбранные.
        if any(len(current&set(old['stems']))/max(1,len(current|set(old['stems'])))>=.8 for old in chosen): continue
        chosen.append(item)
        if len(chosen)==12: break
    return chosen

def summarize(document, corpus, method='textrank', sentence_count=10):  # Построить реферат и вернуть текст, оценки, ключевые фразы и граф.
    started=perf_counter()
    if method not in METHODS: raise ValueError('Неизвестный метод.')
    if not 1<=sentence_count<=50: raise ValueError('Длина реферата: от 1 до 50 предложений.')
    text,language=document['text'],document['language']
    sentences=split_sentences(text)
    if not sentences: raise ValueError('Текст не содержит предложений.')
    if len(sentences)>MAX_SENTENCES:
        raise ValueError(f'Слишком длинный документ: максимум {MAX_SENTENCES} предложений. Загрузите раздел статьи.')
    sc=[counts(s.text,language) for s in sentences]  # Частоты значимых стеммов в каждом предложении.
    dc=counts(text,language)  # Частоты стеммов во всём документе.
    if not dc: raise ValueError('После удаления стоп-слов и чисел не осталось значимых слов.')
    n,idf=corpus_weights(corpus,language)
    if not n: raise ValueError('В коллекции нет документов выбранного языка.')
    max_tf=max(dc.values())
    tw={t:.5*(1+frequency/max_tf)*idf.get(t,0) for t,frequency in dc.items()}  # Модифицированные веса TF-IDF по методичке.
    vectors=sentence_vectors(sc)
    adjacency=sentence_graph(vectors) if method=='textrank' else []  # Базовый метод не использует граф.
    ranks,iterations=pagerank(adjacency) if adjacency else ([],0)
    warnings=[]
    if not any(tw.values()):
        warnings.append('Все IDF равны нулю. Базовые оценки нулевые; равные оценки разрешаются исходным порядком. Добавьте другие документы этого языка.')
    rows=[]
    for i,s in enumerate(sentences):
        posd,posp=s.positions(len(text))  # Множители положения предложения в документе и в абзаце.
        lexical=sum(frequency*tw.get(term,0) for term,frequency in sc[i].items())
        baseline=posd*posp*lexical  # Словесная значимость умножается на оба позиционных множителя.
        score=ranks[i] if method=='textrank' else baseline
        rows.append({'index':i,'text':s.text,'start':s.start,'end':s.end,'paragraph':s.paragraph,
                     'pos_document':posd,'pos_paragraph':posp,'lexical_score':lexical,
                     'baseline_score':baseline,'score':score,'selected':False})
    eligible=[i for i,c in enumerate(sc) if c]  # Предложения без значимых слов не участвуют в отборе.
    if not eligible: raise ValueError('Не найдены информативные предложения.')
    # Внутренняя сортировка выбирает лучшие оценки, внешняя возвращает исходный порядок текста.
    selected=sorted(sorted(eligible,key=lambda i:(-rows[i]['score'],i))[:sentence_count])
    for i in selected: rows[i]['selected']=True
    summary='\n\n'.join(sentences[i].text for i in selected)  # Используются исходные предложения без переписывания.
    keywords=keyword_scores(text,language,method,dc,tw)
    # Для показа берётся часть реального графа; расчёт выполняется по полному графу.
    graph_nodes=sorted(sorted(eligible,key=lambda i:(-rows[i]['score'],i))[:16]) if adjacency else []
    graph_edges=[{'source':i,'target':j,'weight':adjacency[i][j]} for i in graph_nodes for j in graph_nodes if i<j and j in adjacency[i]] if adjacency else []
    redundancy=mean(cosine(vectors[i],vectors[j]) for i,j in combinations(selected,2)) if len(selected)>1 else 0.0  # Среднее сходство выбранных предложений.
    chosen_terms=set(t for i in selected for t in sc[i])
    coverage_weights=tw if any(tw.values()) else dc
    coverage=sum(v for t,v in coverage_weights.items() if t in chosen_terms)/sum(coverage_weights.values())  # Доля веса терминов, покрытых рефератом.
    return {'method':method,'method_label':METHODS[method],'requested_count':sentence_count,
            'selected_count':len(selected),'sentence_count':len(sentences),'selected_indices':selected,
            'summary':summary,'keywords':keywords,'sentences':rows,'warnings':warnings,
            'elapsed_ms':(perf_counter()-started)*1000,'compression':len(summary)/len(text),
            'redundancy':redundancy,'term_coverage':coverage,'iterations':iterations,
            'language_corpus_size':n,'graph_nodes':graph_nodes,'graph_edges':graph_edges,
            'graph_edge_count':sum(len(row) for row in adjacency)//2 if adjacency else 0}
