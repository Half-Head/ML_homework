"""Пример решения первой домашней работы: функции, которые импортирует solution.ipynb.

В примере вместо моделей стоит DummyClassifier: он не смотрит на текст
и показывает, какое качество получается совсем без модели. В вашем решении
функции init_*_models загружают обученные модели из папки artifacts,
а predict_* делают ту обработку текста, на которой модели обучались.
"""

from __future__ import annotations

import csv
import json
import re
import pandas as pd
import numpy as np
import nltk
import os
import joblib
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, f1_score, accuracy_score, balanced_accuracy_score,
                            precision_score, recall_score, confusion_matrix)
from scipy.sparse.csgraph import connected_components 
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors # не классификатор, возвращает соседей


try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

WORD = re.compile(r"[^\W\d_]+")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LANGUAGES = ("english", "russian")
STOPWORDS = {language: set(stopwords.words(language)) for language in LANGUAGES}
STEMMERS = {language: SnowballStemmer(language) for language in LANGUAGES}

SEED = 42
CODE_LINE = re.compile( # для очистки от кусков кода
    r"^\s*(" # ключевые слова только в начале строки
        r"(def|class\s+\w+|import\s+\w|import|elif|return|else|except|with\s+open\(|from\s+\w+\s+import|using\s+namespace|"
        r"lambda|yield|async|await|assert|nonlocal|del|global|pass|break|continue)\b"
        r"|(void|int|long|float|double|char|string|bool|boolean|var|const|"
        r"function|extends|implements|namespace|struct|enum|"
        r"interface|package|override|public|private|protected|static)\b"
        r"|(#include|using\s+namespace)"
        r"|(console\.\w+|System\.out|print\(|printf\(|fmt\.\w+|cout\s*<<)"
        r"|(SELECT|INSERT|JOIN)\b"
    r")"
    # где угодно
    r"|(\bnp|\bnumpy|\bplt|\bpd|\bsns|\bsklearn|\btorch|\btf|\bnn)\.\w+"
    r"|(\b\w+(Error|Exception))\b"
    r"|\bTraceback\b",
    re.IGNORECASE,
)
CODE_BLOCK_PATTERN = re.compile(
    r'("""|\'\'\'|```|~~~)'
    r'.*?' # до первого закрытия
    r'\1', # закрывает та же последовательность 
    re.DOTALL, # соединяет переносы строк
)



# ---------------------------------------------------------------------------
#  Данные
# ---------------------------------------------------------------------------

def read_dialogs(path: str) -> list[dict]:
    """Прочитать jsonl-файл и сложить диалоги в список.

    От каждой строки остаются только номер диалога и реплики:
    {"dialog_id": ..., "turns": [{"role": ..., "text": ...}, ...]}.
    """
    dialogs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                dialogs.append({"dialog_id": record["dialog_id"], "turns": record["turns"]})
    return dialogs


def read_labels(path: str, task: str) -> dict[str, str]:
    """Ответы по задаче task ("reaction" или "topic"): dialog_id -> метка.

    Диалоги, у которых такой метки нет, пропускаются.
    """
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                label = (record.get("labels") or {}).get(task)
                if label is not None:
                    labels[record["dialog_id"]] = label
    return labels


def build_tables(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строит 2 таблицы для удобного просмотра данных.

    Возвращает:
    dialogs_df:
        Одна строка = один диалог. Индекс — dialog_id.
        Колонки: language, n_user_turns, user_text, assistant_text,
        user_len, assistant_len, reaction, task_type.

    turns_df:
        Одна строка = одна реплика. MultiIndex (dialog_id, turn_idx).
        Колонки: role, length, text.
    """

    dialogs, turns = [], []
    for rec in records:
        user_parts, asst_parts = [], []
        for turn_num, turn in enumerate(rec["turns"]):
            turns.append({
                "dialog_id": rec["dialog_id"],
                "turn_idx": turn_num,
                "role": turn["role"],
                "length": len(turn["text"]),
                "text": turn["text"],
            })
            if turn["role"] == "user":
                user_parts.append(turn["text"])
            elif turn["role"] == "assistant":
                asst_parts.append(turn["text"])

        labels = rec.get("labels") or {}
        dialogs.append({
            "dialog_id": rec["dialog_id"],
            "language": rec["language"],
            "n_user_turns": rec["n_user_turns"],
            "user_text": "\n".join(user_parts),
            "assistant_text": "\n".join(asst_parts),
            "user_len": sum(len(part) for part in user_parts),
            "assistant_len": sum(len(part) for part in asst_parts),
            "reaction": labels.get("reaction"),
            "task_type": labels.get("task_type"),
        })

    dialogs_df = pd.DataFrame(dialogs).set_index("dialog_id")
    turns_df = pd.DataFrame(turns).set_index(["dialog_id", "turn_idx"])
    return dialogs_df, turns_df


def get_groups(dialogs, threshold: float = 0.1) -> tuple[int, np.ndarray]:
    """Разбивает диалоги на группы по косинусной близости.

    Возвращает:
    n_gr:
        Число компонент связности (групп).
    groups:
        id группы для каждого диалога.
    """
    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit_transform(dialogs)

    nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(tf_idf) # возвращает 1-cos ! то есть 0 - идентичные, 1 - разные
    distances, indices = nn.kneighbors(tf_idf)
    distances = distances[:, 1:] # убираем расстояние до самого себя, в итоге смотрим 5 соседей каждого по тексту пользователя
    indices = indices[:, 1:]

    rows, cols = [], []
    n = distances.shape[0]

    for first in range(n):
        for j in range(distances.shape[1]):
            if distances[first, j] <= threshold:
                second = indices[first, j]
                rows += [first, second]
                cols += [second, first]

    sparse_matrix = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n,n)) # разреженная матрица смежности
    n_gr, groups = connected_components(sparse_matrix, directed=False) # labels[i] - id группы для i диалога
    return n_gr, groups


# ---------------------------------------------------------------------------
#  Обработка текста, как на семинаре 4
# ---------------------------------------------------------------------------

def normalize(text: str) -> list[str]:
    """Нижний регистр, слова из букв, без стоп-слов, основы слов.

    Язык определяется по доле кириллицы: русский или английский.
    """
    words = WORD.findall(text.lower())
    letters = "".join(words)
    language = "russian" if letters and len(CYRILLIC.findall(letters)) > len(letters) / 2 else "english"
    stop, stemmer = STOPWORDS[language], STEMMERS[language]
    stems = [stemmer.stem(word) for word in words if len(word) >= 2 and word not in stop]
    return stems


def remove_code_blocks(text: str) -> str:
    """Удаляет блоки кода между тройными кавычками"""
    if not isinstance(text, str):
        return ""
    return CODE_BLOCK_PATTERN.sub("", text)


def delete_code_lines(text: str) -> str:
    """Удаляет из текста строки, считающиеся кодом."""
    text = remove_code_blocks(text)
    lines = text.split("\n")
    keep = [line for line in lines if not CODE_LINE.search(line)]
    return "\n".join(keep).strip()


def dialog_text(dialog: dict) -> str:
    """Все реплики пользователя одной строкой из нормализованных слов."""
    user_text = " ".join(turn["text"] for turn in dialog["turns"] if turn["role"] == "user")
    text = " ".join(normalize(user_text))
    return text


def clean_normalize_dialog(
    dialog: dict,
    include_assistant: bool = False,
    clean_code: bool = True,
    num: int | None = None,
    position: str = "first",
) -> str:
    """Все реплики одной строкой из нормализованных слов 

    include_assistant:
        Если True — берутся все реплики (user + assistant).
        Если False — только user-реплики.
    clean_code:
        Удалять ли строки кода через delete_code_lines.
    num:
        Сколько реплик брать. None — все.
        Если include_assistant=True, num умножается на 2
        (то есть по num реплик на каждую роль)!!
    position: {"first", "last"}, default "first"
        Брать первые num реплик или последние.

    """
    if include_assistant:
        text = [turn["text"] for turn in dialog["turns"]]
        if num is not None:
            num *= 2
    else:
        text = [turn["text"] for turn in dialog["turns"] if turn["role"] == "user"]

    if num is not None:
        if position == 'first':
            text = text[:num]
        elif position == 'last':
            text = text[-num:]
        else:
            raise ValueError("position должен быть 'first' или 'last'")
        
    text = "\n".join(text)
    if clean_code:
        text = delete_code_lines(text)
    clean_norm_text = " ".join(normalize(text))
    return clean_norm_text


# ---------------------------------------------------------------------------
#  Модели: реакция пользователя
# ---------------------------------------------------------------------------

def init_reaction_models() -> dict[str, DummyClassifier]:
    """Все модели задачи 1: имя -> готовая к предсказанию модель.
    """
    artifacts = Path("artifacts") / "reactions"
    models = {
        "dummy_most_frequent": joblib.load(artifacts / "dmf.joblib"),
        "dummy_stratified": joblib.load(artifacts / "ds.joblib"),
        "complement_nb": joblib.load(artifacts / "nb.joblib"),
        "log_reg": joblib.load(artifacts / "log_reg.joblib"),
        "linear_svc": joblib.load(artifacts / "svc.joblib"),
        "my_model": joblib.load(artifacts / "mymodel.joblib"),
    }
    return models


def predict_reaction(model, dialogs: list[dict]) -> list[str]:
    """По одному ответу на диалог: "positive", "neutral" или "negative"."""
    if isinstance(model, dict) and "model" in model and "vectorizer" in model:
        texts = [clean_normalize_dialog(d) for d in dialogs]
        texts_tfidf = model["vectorizer"].transform(texts)
        return [str(label) for label in model["model"].predict(texts_tfidf)]

    #для своей модели
    return [str(label) for label in model.predict(dialogs)]



# ---------------------------------------------------------------------------
#  Модели: темы
# ---------------------------------------------------------------------------

def init_topic_models(train_path: str) -> dict[str, DummyClassifier]:
    """Все модели задачи 2: имя -> готовая к предсказанию модель.

    Меток тем в данных нет, темы придумываете вы. В примере вместо ваших тем
    две заглушки: одна тема на все диалоги и десять случайных тем.
    """
    dialog_ids = [dialog["dialog_id"] for dialog in read_dialogs(train_path)]
    one_topic = ["все диалоги"] * len(dialog_ids)
    ten_topics = [f"тема {k % 10}" for k in range(len(dialog_ids))]
    models = {
        "dummy_one_topic": DummyClassifier(strategy="most_frequent").fit(dialog_ids, one_topic),
        "dummy_random_topics": DummyClassifier(strategy="uniform", random_state=0).fit(dialog_ids, ten_topics),
    }
    return models


def predict_topic(model, dialogs: list[dict]) -> list[str]:
    """По одному ответу на диалог: название темы."""
    texts = [dialog_text(dialog) for dialog in dialogs]
    answers = [str(label) for label in model.predict(texts)]
    return answers


def _training_data(train_path: str, task: str) -> tuple[list[str], list[str]]:
    labels = read_labels(train_path, task)
    dialogs = [dialog for dialog in read_dialogs(train_path) if dialog["dialog_id"] in labels]
    texts = [dialog_text(dialog) for dialog in dialogs]
    targets = [labels[dialog["dialog_id"]] for dialog in dialogs]
    return texts, targets


# ---------------------------------------------------------------------------
#  Метрики и сохранение ответов
# ---------------------------------------------------------------------------

def compute_metrics(dialogs: list[dict], predictions: list[str], labels: dict[str, str]) -> dict[str, float]:
    """Метрики по тем диалогам, для которых известен ответ."""
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    y_true = [true for true, _ in pairs]
    y_pred = [pred for _, pred in pairs]
    metrics = {"accuracy": accuracy_score(y_true, y_pred)}
    return metrics


def compute_partition_metrics(dialogs: list[dict], predictions: list[str], labels: dict[str, str]) -> dict[str, float]:
    """Сравнение разбиений для тем: названия тем не обязаны совпадать,
    сравнивается, какие диалоги попали в одну тему."""
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    y_true = [true for true, _ in pairs]
    y_pred = [pred for _, pred in pairs]
    metrics = {"ARI": adjusted_rand_score(y_true, y_pred)}
    return metrics


def save_predictions(path: str, dialogs: list[dict], predictions: dict[str, dict[str, list[str]]]) -> None:
    """Сохранить ответы всех моделей: dialog_id, task, model, prediction."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for task, by_model in predictions.items():
            for name, answers in by_model.items():
                for dialog, answer in zip(dialogs, answers):
                    writer.writerow([dialog["dialog_id"], task, name, answer])


def calculate_metrics(y_true, y_pred, labels: list[str]) -> dict:
    """Считает метрики
    y_true: Истинные значения
    y_pred: Предсказание
    labels: Cписок меток в нужном порядке
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro-f1": f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        "weighted-f1": f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels)
    }

    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_metrics = {}
    for i in range(len(labels)):
        per_class_metrics[f"precision_{labels[i]}"] = precision[i]
        per_class_metrics[f"recall_{labels[i]}"] = recall[i]
        per_class_metrics[f"f1_{labels[i]}"] = f1[i]
    metrics["per_class_metrics"] = per_class_metrics
    
    return metrics


def vectorize_topic_texts(user_texts, *, vectorizer=None, min_df=2, max_df=0.95, max_features=30_000):
    """TF-IDF для диалогов, представленных списками реплик пользователя.

    user_texts[i] — список или массив строк одного диалога.
    Если vectorizer не передан, обучает новый.
    Если передан, только преобразует тексты его словарём.

    Возвращает:
        X — разреженная матрица (число диалогов, число признаков);
        vectorizer — обученный TfidfVectorizer.
    """
    documents = [
        "\n".join(parts)
        for parts in user_texts
    ]

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            tokenizer=normalize,
            token_pattern=None,
            lowercase=False,  # normalize уже приводит к нижнему регистру
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
        )
        X = vectorizer.fit_transform(documents)
    else:
        X = vectorizer.transform(documents)

    return X, vectorizer


# ---------------------------------------------------------------------------
#  Сохранение моделей
# ---------------------------------------------------------------------------


def save_model(model, path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(model, path)
    print(f"сохранено: {path}, {os.path.getsize(path) / 1024:.0f} КБ")


# ---------------------------------------------------------------------------
#  Свои модели
# ---------------------------------------------------------------------------


class MyModelReactions:
    def __init__(self, weight=0.4, C=0.1):
        self.weight = weight
        self.C = C

        self.vectorizer_full = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.vectorizer_last2 = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=6)

        self.model_full = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )
        self.model_last2 = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )

        self.classes_ = None
        self.is_fitted_ = False

    def fit(self, dialogs, y):
        full_texts = [clean_normalize_dialog(d) for d in dialogs]
        last2_texts = [clean_normalize_dialog(d, num=2, position='last') for d in dialogs]

        X_full = self.vectorizer_full.fit_transform(full_texts)
        X_last2 = self.vectorizer_last2.fit_transform(last2_texts)

        self.model_full.fit(X_full, y)
        self.model_last2.fit(X_last2, y)

        self.classes_ = self.model_full.classes_
        self.is_fitted_ = True

        return self

    def predict_proba(self, dialogs):
        if not self.is_fitted_:
            raise RuntimeError("Сначала fit()")

        full_texts = [clean_normalize_dialog(d) for d in dialogs]
        last2_texts = [clean_normalize_dialog(d, num=2, position='last') for d in dialogs]

        X_full = self.vectorizer_full.transform(full_texts)
        X_last2 = self.vectorizer_last2.transform(last2_texts)

        p_full = self.model_full.predict_proba(X_full)
        p_last2 = self.model_last2.predict_proba(X_last2)

        return self.weight * p_last2 + (1 - self.weight) * p_full

    def predict(self, dialogs):
        proba = self.predict_proba(dialogs)
        return self.classes_[proba.argmax(axis=1)]