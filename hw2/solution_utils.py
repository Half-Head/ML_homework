"""Пример решения второй домашней работы: функции, которые импортирует solution.ipynb.

Задача одна — реакция пользователя. Модели делятся на две группы по префиксу
имени: `hw1_` — модели первой домашней работы, `hw2_` — новые, на деревьях
и ансамблях. В примере модели обучаются прямо в init_reaction_models, чтобы
пример запускался без папки artifacts. В вашем решении init_reaction_models
загружает обученные модели из artifacts, а predict_reaction делает ту обработку
текста, на которой модели обучались.
"""

from __future__ import annotations

import csv
import json
import re
import os
import sys
import importlib.util
from pathlib import Path
from sklearn.ensemble import StackingClassifier
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.metrics import (adjusted_rand_score, f1_score, accuracy_score, balanced_accuracy_score,
                            precision_score, recall_score, confusion_matrix)
from scipy.sparse.csgraph import connected_components 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

SEED = 42

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

WORD = re.compile(r"[^\W\d_]+")
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LANGUAGES = ("english", "russian")
STOPWORDS = {language: set(stopwords.words(language)) for language in LANGUAGES}
STEMMERS = {language: SnowballStemmer(language) for language in LANGUAGES}
REACTIONS = ["negative", "neutral", "positive"]
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


def dialog_text(dialog: dict) -> str:
    """Все реплики пользователя одной строкой из нормализованных слов."""
    user_text = " ".join(turn["text"] for turn in dialog["turns"] if turn["role"] == "user")
    text = " ".join(normalize(user_text))
    return text


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
#  Модели: hw1_ — старые, hw2_ — новые
# ---------------------------------------------------------------------------

def init_reaction_models(train_path=None):
    base_path = Path(__file__).resolve().parent
    hw1_path = base_path.parent / "hw1"

    # Исходный класс нужен для загрузки старой MyModel.
    spec = importlib.util.spec_from_file_location(
        "hw1_solution_utils", hw1_path / "solution_utils.py"
    )
    hw1_utils = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hw1_utils
    spec.loader.exec_module(hw1_utils)

    models = {}
    current_utils = sys.modules["solution_utils"]
    try:
        sys.modules["solution_utils"] = hw1_utils
        for name in ["log_reg", "logreg_balanced", "mymodel", "svc", "dmf", "ds", "nb"]:
            models[f"hw1_{name}"] = joblib.load(
                hw1_path / "artifacts/reactions" / f"{name}.joblib"
            )
    finally:
        sys.modules["solution_utils"] = current_utils

    for name in [
        "bagging", "random_forest", "lightgbm", "catboost",
        "stacking", "mymodel_lightgbm_stacking",
    ]:
        models[f"hw2_{name}"] = joblib.load(
            base_path / "artifacts" / f"{name}.joblib"
        )

    return models


def predict_reaction(saved, dialogs):
    if not isinstance(saved, dict):
        answers = saved.predict(dialogs)
    else:
        model = saved["model"]

        if isinstance(model, MyModelReactions):
            answers = model.predict(dialogs)
        else:
            texts = [clean_normalize_dialog(d) for d in dialogs]

            if isinstance(model, StackingClassifier):
                answers = model.predict(texts)
            else:
                features = saved["vectorizer"].transform(texts)
                answers = model.predict(features)

    return np.asarray(answers).ravel().astype(str).tolist()


def _training_data(train_path: str) -> tuple[list[str], list[str]]:
    labels = read_labels(train_path)
    dialogs = [dialog for dialog in read_dialogs(train_path) if dialog["dialog_id"] in labels]
    texts = [dialog_text(dialog) for dialog in dialogs]
    targets = [labels[dialog["dialog_id"]] for dialog in dialogs]
    data = (texts, targets)
    return data


# ---------------------------------------------------------------------------
#  Метрики и сохранение ответов
# ---------------------------------------------------------------------------

def compute_metrics(dialogs: list[dict], predictions: list[str], labels: dict[str, str]) -> dict[str, float]:
    """macro-F1 и accuracy по тем диалогам, для которых известен ответ."""
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    y_true = [true for true, _ in pairs]
    y_pred = [pred for _, pred in pairs]
    metrics = {
        "macro-F1": f1_score(y_true, y_pred, labels=REACTIONS, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    return metrics


def save_predictions(path: str, dialogs: list[dict], predictions: dict[str, list[str]]) -> None:
    """Сохранить ответы всех моделей: dialog_id, task, model, prediction."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for name, answers in predictions.items():
            for dialog, answer in zip(dialogs, answers):
                writer.writerow([dialog["dialog_id"], "reaction", name, answer])


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


def save_model(model, path: str) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(model, path)
    print(f"сохранено: {path}, {os.path.getsize(path) / 1024:.0f} КБ")

# ---------------------------------------------------------------------------
# Свои модели
# ---------------------------------------------------------------------------

class MyModelReactions(ClassifierMixin, BaseEstimator):
    def __init__(self, weight, tfidf, model_full, model_last2):
        self.weight = weight
        self.tfidf = tfidf
        self.model_full = model_full
        self.model_last2 = model_last2

    def fit(self, dialogs, y):
        self.vectorizer_full = clone(self.tfidf)
        self.vectorizer_last2 = clone(self.tfidf)
        self.model_full_ = clone(self.model_full)
        self.model_last2_ = clone(self.model_last2)

        full_texts = [clean_normalize_dialog(d) for d in dialogs]
        last2_texts = [
            clean_normalize_dialog(d, num=2, position="last")
            for d in dialogs
        ]

        X_full = self.vectorizer_full.fit_transform(full_texts)
        X_last2 = self.vectorizer_last2.fit_transform(last2_texts)

        self.model_full_.fit(X_full, y)
        self.model_last2_.fit(X_last2, y)

        self.classes_ = self.model_full_.classes_
        return self

    def predict_proba(self, dialogs):
        full_texts = [clean_normalize_dialog(d) for d in dialogs]
        last2_texts = [
            clean_normalize_dialog(d, num=2, position="last")
            for d in dialogs
        ]

        X_full = self.vectorizer_full.transform(full_texts)
        X_last2 = self.vectorizer_last2.transform(last2_texts)

        p_full = self.model_full_.predict_proba(X_full)
        p_last2 = self.model_last2_.predict_proba(X_last2)

        return self.weight * p_last2 + (1 - self.weight) * p_full

    def predict(self, dialogs):
        proba = self.predict_proba(dialogs)
        return self.classes_[proba.argmax(axis=1)]