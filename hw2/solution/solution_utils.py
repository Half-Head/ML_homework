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

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

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


def read_labels(path: str) -> dict[str, str]:
    """Реакция пользователя: dialog_id -> "positive", "neutral" или "negative".

    Диалоги без метки реакции пропускаются.
    """
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                label = (record.get("labels") or {}).get("reaction")
                if label is not None:
                    labels[record["dialog_id"]] = label
    return labels


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


# ---------------------------------------------------------------------------
#  Модели: hw1_ — старые, hw2_ — новые
# ---------------------------------------------------------------------------

def init_reaction_models(train_path: str) -> dict:
    """Все модели: имя -> готовая к предсказанию модель.

    Имена старых моделей начинаются с hw1_, новых — с hw2_: по префиксу
    модели делятся на группы при проверке. В примере модели обучаются здесь
    на размеченных диалогах из train_path; в вашем решении здесь загрузка
    из artifacts.
    """
    texts, labels = _training_data(train_path)
    models = {
        "hw1_dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "hw1_tfidf_logreg": make_pipeline(
            TfidfVectorizer(min_df=2),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "hw2_tfidf_random_forest": make_pipeline(
            TfidfVectorizer(min_df=2, max_features=5000),
            RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0),
        ),
        "hw2_tfidf_svd_boosting": make_pipeline(
            TfidfVectorizer(min_df=2),
            TruncatedSVD(n_components=100, random_state=0),
            HistGradientBoostingClassifier(class_weight="balanced", random_state=0),
        ),
    }
    for model in models.values():
        model.fit(texts, labels)
    return models


def predict_reaction(model, dialogs: list[dict]) -> list[str]:
    """По одному ответу на диалог: "positive", "neutral" или "negative"."""
    texts = [dialog_text(dialog) for dialog in dialogs]
    answers = [str(label) for label in model.predict(texts)]
    return answers


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
