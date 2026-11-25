import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import db
from .search import Search

CHART_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "static", "charts")

ELEVEN_LEVELS = [round(i / 10, 1) for i in range(11)]


def evaluate_query(query_id, query_text):
    search = Search(all_words_together=True)
    search.search_query = query_text
    results = search.GetSearchResult()
    ranked = [r.document_id for r in results]
    relevant = db.get_relevant_ids(query_id)
    n_docs = db.total_documents()

    n_rel = len(relevant)
    hits = 0
    precision_sum_at_rel = 0.0
    pr_points = []          # (recall, precision) в позициях релевантных документов
    ranks_of_rel = []

    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            hits += 1
            p_i = hits / i
            r_i = hits / n_rel if n_rel else 0.0
            precision_sum_at_rel += p_i
            pr_points.append((r_i, p_i))
            ranks_of_rel.append(i)

    n_ret = len(ranked)
    precision = hits / n_ret if n_ret else 0.0
    recall = hits / n_rel if n_rel else 0.0
    fallout = (n_ret - hits) / (n_docs - n_rel) if n_docs > n_rel else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    r_precision = 0.0
    if n_rel:
        top_r = ranked[:n_rel]
        r_precision = (len(set(top_r) & relevant) / n_rel) if top_r else 0.0

    p_at_5 = (len(set(ranked[:5]) & relevant) / 5) if ranked[:5] else 0.0
    p_at_10 = (len(set(ranked[:10]) & relevant) / 10) if ranked[:10] else 0.0

    ap = precision_sum_at_rel / n_rel if n_rel else 0.0

    # 11-точечная интерполяция для этого запроса
    eleven = []
    for level in ELEVEN_LEVELS:
        candidates = [p for (r, p) in pr_points if r >= level - 1e-9]
        eleven.append(max(candidates) if candidates else 0.0)

    return {
        "query_id": query_id,
        "query_text": query_text,
        "n_relevant": n_rel,
        "n_returned": n_ret,
        "n_found": hits,
        "precision": precision,
        "recall": recall,
        "fallout": fallout,
        "f1": f1,
        "r_precision": r_precision,
        "p_at_5": p_at_5,
        "p_at_10": p_at_10,
        "ap": ap,
        "pr_points": pr_points,
        "eleven": eleven,
    }


def evaluate_all():
    per_query = []
    for q in db.get_test_queries():
        per_query.append(evaluate_query(q["query_id"], q["query_text"]))

    n = len(per_query)
    averages = {
        "map": sum(q["ap"] for q in per_query) / n if n else 0.0,
        "precision": sum(q["precision"] for q in per_query) / n if n else 0.0,
        "recall": sum(q["recall"] for q in per_query) / n if n else 0.0,
        "f1": sum(q["f1"] for q in per_query) / n if n else 0.0,
        "r_precision": sum(q["r_precision"] for q in per_query) / n if n else 0.0,
        "fallout": sum(q["fallout"] for q in per_query) / n if n else 0.0,
        "p_at_5": sum(q["p_at_5"] for q in per_query) / n if n else 0.0,
        "p_at_10": sum(q["p_at_10"] for q in per_query) / n if n else 0.0,
    }
    eleven_avg = [
        sum(q["eleven"][i] for q in per_query) / n if n else 0.0
        for i in range(len(ELEVEN_LEVELS))
    ]
    return {"per_query": per_query, "averages": averages, "eleven_avg": eleven_avg}


def _setup_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def render_charts(data):
    _setup_style()
    os.makedirs(CHART_DIR, exist_ok=True)
    per_query = data["per_query"]
    labels = [f"З{i + 1}" for i in range(len(per_query))]

    # 1. Гистограмма P / R / F1 по запросам
    if per_query:
        x = range(len(per_query))
        width = 0.26
        fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
        ax.bar([i - width for i in x], [q["precision"] for q in per_query],
               width, label="Точность (P)", color="#2563eb")
        ax.bar(list(x), [q["recall"] for q in per_query],
               width, label="Полнота (R)", color="#16a34a")
        ax.bar([i + width for i in x], [q["f1"] for q in per_query],
               width, label="F1-мера", color="#f59e0b")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Значение метрики")
        ax.set_xlabel("Тестовые запросы")
        ax.set_title("Точность, полнота и F1-мера по тестовым запросам")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
        fig.savefig(os.path.join(CHART_DIR, "bar_metrics.png"), dpi=150)
        plt.close(fig)

        # 2. Кривые полнота–точность
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        for i, q in enumerate(per_query):
            if q["pr_points"]:
                rs = [pt[0] for pt in q["pr_points"]]
                ps = [pt[1] for pt in q["pr_points"]]
                ax.plot(rs, ps, marker="o", markersize=3.5,
                        linewidth=1.4, label=labels[i])
        ax.set_xlabel("Полнота (Recall)")
        ax.set_ylabel("Точность (Precision)")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.08)
        ax.set_title("Кривые полнота–точность по тестовым запросам")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.savefig(os.path.join(CHART_DIR, "pr_curves.png"), dpi=150)
        plt.close(fig)

    # 3. 11-точечная интерполированная средняя точность
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.plot(ELEVEN_LEVELS, data["eleven_avg"], marker="s", color="#2563eb",
            linewidth=2, label="Средняя по коллекции")
    for i, q in enumerate(data["per_query"]):
        ax.plot(ELEVEN_LEVELS, q["eleven"], linewidth=0.8, alpha=0.35,
                label=None)
    ax.set_xlabel("Уровень полноты")
    ax.set_ylabel("Интерполированная точность")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("11-точечная интерполированная средняя точность")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(CHART_DIR, "eleven_point.png"), dpi=150)
    plt.close(fig)

    return ["bar_metrics.png", "pr_curves.png", "eleven_point.png"]
