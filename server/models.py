import numpy as np

def compute_tf(collection):
    """Вычисляет TF матрицу"""
    unique_words = set([word for document in collection for word in document])
    unique_words = list(unique_words)

    index_words = {}
    for index, word in enumerate(unique_words):
        index_words[word] = index

    n = len(collection)
    m = len(unique_words)
    tf = np.zeros(shape=(n, m))

    for index, document in enumerate(collection):
        total_words = len(document)
        unique_in_doc = set(document)
        for word in unique_in_doc:
            tf[index, index_words[word]] = document.count(word) / total_words

    return tf, index_words

def compute_idf(collection, index_words):
    """Вычисляет IDF матрицу"""
    m = len(index_words)
    n = len(collection)
    idf = np.zeros(m)

    count = dict.fromkeys(index_words, 0)
    for document in collection:
        for word in set(document):
            count[word] += 1

    for word, index in index_words.items():
        idf[index] = np.log(n / count[word])

    return idf

def compute_tfidf(tf_matrix, idf_matrix):
    """Вычисляет TF-IDF матрицу"""
    tfidf = tf_matrix * idf_matrix
    return tfidf

def compute_bag_of_words(collection):
    """Вычисляет метрику Мешок Слов"""
    unique_words = set([word for document in collection for word in document])
    unique_words = list(unique_words)

    index_words = {}
    for index, word in enumerate(unique_words):
        index_words[word] = index

    n = len(collection)
    m = len(unique_words)
    bag_of_words = np.zeros(shape=(n, m))

    for index, document in enumerate(collection):
        for word in document:
            bag_of_words[index, index_words[word]] += 1

    return bag_of_words
