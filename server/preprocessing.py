import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
import pymorphy3
from natasha import NewsNERTagger, NewsEmbedding, Segmenter, Doc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt_tab')


def tokenize(text):
    """Токенизация"""
    sentences = sent_tokenize(text, language='russian')
    tokens = word_tokenize(text, language='russian')
    return sentences, tokens


def stemming(texts):
    """Стемминг (грубое приведение к начальной форме)"""
    stemmer = SnowballStemmer('russian')

    if isinstance(texts, str):
        words = word_tokenize(texts, language='russian')
        return [stemmer.stem(word) for word in words]

    results = []
    for text in texts:
        words = word_tokenize(text, language='russian')
        stems = [stemmer.stem(word) for word in words]
        results.append(stems)

    return results

def lemmatize(text):
    """Лемматизация (точное приведение к нормальной форме)"""
    morph = pymorphy3.MorphAnalyzer()
    tokens = word_tokenize(text, language='russian')

    lemmas = []
    for token in tokens:
        if token.isalpha():
            parsed = morph.parse(token)[0]
            lemmas.append((token, parsed.normal_form))
        else:
            lemmas.append((token, token))

    return lemmas

def pos_tagging(text):
    """Part-of-Speech (POS) tagging"""
    morph = pymorphy3.MorphAnalyzer()
    tokens = word_tokenize(text, language='russian')
    russian_pos = {
        'NOUN': 'СУЩ',
        'ADJF': 'ПРИЛ',
        'ADJS': 'ПРИЛ_КР',
        'VERB': 'ГЛАГ',
        'INFN': 'ИНФ',
        'PRTF': 'ПРИЧ',
        'PRTS': 'ПРИЧ_КР',
        'GRND': 'ДЕЕПР',
        'NUMR': 'ЧИСЛ',
        'ADVB': 'НАРЕЧ',
        'NPRO': 'МЕСТ',
        'PREP': 'ПРЕДЛ',
        'CONJ': 'СОЮЗ',
        'PRCL': 'ЧАСТ',
        'INTJ': 'МЕЖД',
    }

    tagged = []
    for token in tokens:
        if not token.isalpha():
            tagged.append((token, 'ПУНКТ'))
        else:
            parsed = morph.parse(token)[0]
            pos = str(parsed.tag.POS) if parsed.tag.POS else 'X'
            pos_display = russian_pos.get(pos, pos)
            tagged.append((token, pos_display))

    return tagged

def entity_recognition(text):
    """Извлечение именованных сущностей (Named Entity Recognition - NER)"""
    segmenter = Segmenter()
    emb = NewsEmbedding()
    ner_tagger = NewsNERTagger(emb)

    document = Doc(text)
    document.segment(segmenter)
    document.tag_ner(ner_tagger)

    ner = []
    for span in document.spans:
        ner.append({
            'текст': span.text,
            'название сущности': span.type
        })

    return ner

def lsa_analyzer(texts):
    """Латентный семантический анализ (LSA)"""
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(texts)

    max_topics = min(tfidf_matrix.shape) - 1
    if max_topics < 1:
        max_topics = 1

    svd = TruncatedSVD(n_components=max_topics)
    lsa = svd.fit_transform(tfidf_matrix)

    return lsa, svd

def word2vec_vectorize(documents, vector_size=100, window=5, min_count=1):
    """Векторизация с помощью Word2Vec"""
    tokenized_docs = []
    for document in documents:
        tokens = word_tokenize(document.lower(), language='russian')
        clean_tokens = [t for t in tokens if t.isalpha()]
        tokenized_docs.append(clean_tokens)

    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,
        epochs=10
    )

    word_vectors = {}
    for word in model.wv.index_to_key:
        word_vectors[word] = model.wv[word].tolist()

    return word_vectors, tokenized_docs