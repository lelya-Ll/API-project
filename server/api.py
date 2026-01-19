from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import models
import preprocessing

app = FastAPI(title='Анализатор текстовых данных')

@app.get("/")
def api_status():
    """Проверка, что API работает"""
    return {"status": "API работает", "message": "Связь установлена"}

@app.post("/tf-idf")
def get_all_tfidf(texts: List[str]):
    """Получить все TF-IDF значения для всех документов коллекции"""
    if not texts:
        raise HTTPException(400, "Нужно передать тексты для анализа")

    documents = [text.split() for text in texts]

    tf_matrix, index_words = models.compute_tf(documents)
    idf_matrix = models.compute_idf(documents, index_words)
    tfidf_matrix = models.compute_tfidf(tf_matrix, idf_matrix)

    formatted_results = {}
    for doc_id in range(len(documents)):
        formatted_results[f"document_{doc_id}"] = {
            "words": documents[doc_id],
            "tf-idf_matrix_row": tfidf_matrix[doc_id].tolist()
        }

    return {
        "total_documents": len(documents),
        "results": formatted_results,
        "full_matrix": tfidf_matrix.tolist()
    }

@app.post("/bag-of-words")
def get_bag_of_words(texts: List[str]):
    """Получить мешок слов (Bag of words)"""
    if not texts:
        raise HTTPException(400, "Нужно передать тексты для анализа")

    documents = [text.split() for text in texts]
    bow_matrix = models.compute_bag_of_words(documents)

    return {
        "matrix": bow_matrix.tolist(),
        "shape": bow_matrix.shape
    }

@app.post("/text_nltk/tokenize")
def nltk_tokenize(text: str):
    """Токенизация текста"""
    if not text:
        raise HTTPException(400, "Нужно передать текст для токенизации")

    sentences, tokens = preprocessing.tokenize(text)
    return {
        "sentences": sentences,
        "tokens": tokens
    }

@app.post("/text_nltk/stemming")
def nltk_stemming(text: str):
    """Стемминг текста"""
    if not text:
        raise HTTPException(400, "Нужно передать текст для стемминга")

    words = text.split()
    stems = preprocessing.stemming([words])
    return {
        "original": words,
        "stems": stems
    }

@app.post("/text_nltk/lemmatize")
def nltk_lemmatize(text: str):
    """Лемматизация текста"""
    if not text:
        raise HTTPException(400, "Нужно передать текст для лемматизации")

    lemmas = preprocessing.lemmatize(text)
    return {
        "lemmas": lemmas
    }

@app.post("/text_nltk/pos")
def nltk_pos(text: str):
    """Part-of-Speech tagging"""
    if not text:
        raise HTTPException(400, "Нужно передать текст для POS-разметки")

    tags = preprocessing.pos_tagging(text)
    return {
        "pos_tags": tags
    }

@app.post("/text_nltk/ner")
def nltk_ner(text: str):
    """Named Entity Recognition"""
    if not text:
        raise HTTPException(400, "Нужно передать текст для NER")

    entities = preprocessing.entity_recognition(text)
    return {
        "entities": entities
    }

@app.post("/lsa")
def lsa_analysis(texts: List[str]):
    """Латентный семантический анализ"""
    if not texts:
        raise HTTPException(400, "Нужно передать тексты для LSA")

    lsa_matrix, svd_model = preprocessing.lsa_analyzer(texts)

    return {
        "lsa_matrix": lsa_matrix.tolist(),
        "explained_variance": svd_model.explained_variance_ratio_.tolist()
    }

@app.post("/word2vec")
def word2vec_analysis(texts: List[str]):
    """Word2Vec векторное представление"""
    if not texts:
        raise HTTPException(400, "Нужно передать тексты для Word2Vec")

    word_vectors, tokenized = preprocessing.word2vec_vectorize(texts)

    sample_words = list(word_vectors.keys())[:5]

    return {
        "vocabulary_size": len(word_vectors),
        "sample_vectors": {word: word_vectors[word] for word in sample_words}
    }

@app.post("/upload-and-process")
def upload_and_process(file: UploadFile = File(...), method: str = "tf-idf"):
    if not file.filename.endswith('.txt'):
        raise HTTPException(400, "Разрешены только .txt файлы")

    content = file.file.read().decode("utf-8")
    texts = [line.strip() for line in content.split("\n") if line.strip()]

    if not texts:
        raise HTTPException(400, "Файл пустой")

    base_result = {
        "filename": file.filename,
        "method": method,
        "lines_count": len(texts)
    }

    if method == "tf-idf":
        analysis_result = get_all_tfidf(texts)
    elif method == "bag-of-words":
        analysis_result = get_bag_of_words(texts)
    elif method == "tokenize":
        results = []
        for i, text in enumerate(texts[:5]):
            sentences, tokens = preprocessing.tokenize(text)
            results.append({
                "line": i + 1,
                "text": text,
                "sentences": sentences,
                "tokens": tokens,
                "tokens_count": len(tokens)
            })
        analysis_result = {
            "processed_lines": len(results),
            "results": results
        }
    elif method == "stemming":
        results = []
        for i, text in enumerate(texts[:5]):
            stems = preprocessing.stemming([text])
            results.append({
                "line": i + 1,
                "text": text,
                "stems": stems[0] if stems else [],
                "stems_count": len(stems[0]) if stems else 0
            })
        analysis_result = {
            "processed_lines": len(results),
            "results": results
        }
    elif method == "lemmatize":
        results = []
        for i, text in enumerate(texts[:5]):
            lemmas = preprocessing.lemmatize(text)
            results.append({
                "line": i + 1,
                "text": text,
                "lemmas": lemmas,
                "lemmas_count": len(lemmas)
            })
        analysis_result = {
            "processed_lines": len(results),
            "results": results
        }
    elif method == "pos":
        results = []
        for i, text in enumerate(texts[:5]):
            pos_tags = preprocessing.pos_tagging(text)
            results.append({
                "line": i + 1,
                "text": text,
                "pos_tags": pos_tags,
                "tags_count": len(pos_tags)
            })
        analysis_result = {
            "processed_lines": len(results),
            "results": results
        }
    elif method == "ner":
        results = []
        entities_found = 0
        for i, text in enumerate(texts[:5]):
            entities = preprocessing.entity_recognition(text)
            has_entities = len(entities) > 0
            if has_entities:
                entities_found += 1
            results.append({
                "line": i + 1,
                "text": text,
                "entities": entities,
                "entities_count": len(entities),
                "has_entities": has_entities
            })
        analysis_result = {
            "processed_lines": len(results),
            "entities_found_in": entities_found,
            "results": results
        }
    elif method == "lsa":
        analysis_result = lsa_analysis(texts)
    elif method == "word2vec":
        analysis_result = word2vec_analysis(texts)
    else:
        return {"error": f"Неизвестный метод: {method}", "available_methods": [
            "tf-idf", "bag-of-words", "tokenize", "stemming", "lemmatize",
            "pos", "ner", "lsa", "word2vec"
        ]}

    return {**base_result, **analysis_result}