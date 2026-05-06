import spacy
import nltk
from nltk.corpus import wordnet as wn

nlp = None

def get_nlp():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "Модель en_core_web_sm не найдена. Установите: python -m spacy download en_core_web_sm"
            )
    return nlp

def get_wordnet_data(lemma):
    synsets = wn.synsets(lemma)
    data = []
    for syn in synsets[:3]:  # ограничиваем первыми 3 синсетами
        data.append({
            "name": syn.name(),
            "pos": syn.pos(),
            "definition": syn.definition(),
            "examples": syn.examples()[:2],
            "lemmas": [l.name() for l in syn.lemmas()][:5]
        })
    return data

def analyze_text(text):
    doc = get_nlp()(text)
    sentences = []

    for sent in doc.sents:
        sent_data = {
            "text": sent.text.strip(),
            "tokens": []
        }
        for token in sent:
            tok_data = {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "head_text": token.head.text,
                "head_pos": token.head.pos_,
                "morph": str(token.morph),
                "entity_type": None,
                "entity_label": None,
                "wordnet": []
            }
            if token.ent_type_:
                tok_data["entity_type"] = token.ent_type_
                tok_data["entity_label"] = token.ent_iob_

            if token.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and token.lemma_.isalpha():
                tok_data["wordnet"] = get_wordnet_data(token.lemma_)

            sent_data["tokens"].append(tok_data)
        sentences.append(sent_data)

    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    return {
        "sentences": sentences,
        "entities": entities
    }
