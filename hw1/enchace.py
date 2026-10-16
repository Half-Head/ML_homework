from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from solution_utils import read_dialogs, find_duplicate_groups, vectorize_topic_texts


DATA_PATH = 'hw1_train.jsonl'

TRAIN_PATH = 'train_split_ver2.txt'
VAL_PATH = 'val_split_ver2.txt'
TEST_PATH = 'test_split_ver2.txt'

C = 1
CLASS_WEIGHT = 'balanced'

MIN_DF = 2
MAX_DF = 0.95
MAX_FEATURES = 30_000
NGRAM_RANGE = (1, 2)
FULL_NORMALIZE = True

CONFIDENCE_THRESHOLD = 0.90

MANUAL_GROUP_FILES = ['funfic_id.txt']

def read_labels(path):
    labels = {}

    for line in Path(path).read_text(
        encoding='utf-8-sig'
    ).splitlines():
        if line.strip():
            dialog_id, label = line.split()
            labels[dialog_id] = int(label)

    return labels


def save_labels(path, labels):
    text = '\n'.join(
        f'{dialog_id} {label}'
        for dialog_id, label in labels.items()
    )
    Path(path).write_text(text + '\n', encoding='utf-8')


def user_turns(dialog):
    return [
        turn['text']
        for turn in dialog['turns']
        if turn['role'] == 'user'
    ]


def main():
    train_labels = read_labels(TRAIN_PATH)
    val_labels = read_labels(VAL_PATH)
    test_labels = read_labels(TEST_PATH)

    dialogs = read_dialogs(DATA_PATH)
    dialogs_by_id = {
        dialog['dialog_id']: dialog
        for dialog in dialogs
    }

    all_ids = [dialog['dialog_id'] for dialog in dialogs]
    all_texts = [user_turns(dialog) for dialog in dialogs]

    groups = np.asarray(find_duplicate_groups(
        ['\n'.join(parts) for parts in all_texts],
        threshold=0.1,
    )).copy()

    for path in MANUAL_GROUP_FILES:
        manual_ids = set(
            Path(path).read_text(
                encoding='utf-8-sig'
            ).split()
        )

        manual_mask = np.array([
            dialog_id in manual_ids
            for dialog_id in all_ids
        ])

        touched_groups = np.unique(groups[manual_mask])

        if len(touched_groups):
            groups[np.isin(groups, touched_groups)] = (
                groups.max() + 1
            )

    group_by_id = dict(zip(all_ids, groups))

    heldout_ids = set(val_labels) | set(test_labels)
    blocked_groups = {
        group_by_id[dialog_id]
        for dialog_id in heldout_ids
    }

    used_ids = set(train_labels) | heldout_ids

    candidate_ids = [
        dialog_id
        for dialog_id in all_ids
        if dialog_id not in used_ids
        and group_by_id[dialog_id] not in blocked_groups
    ]

    train_ids = list(train_labels)

    train_texts = [
        user_turns(dialogs_by_id[dialog_id])
        for dialog_id in train_ids
    ]
    y_train = np.array([
        train_labels[dialog_id]
        for dialog_id in train_ids
    ])

    X_train, vectorizer = vectorize_topic_texts(
        train_texts,
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        full_normalize=FULL_NORMALIZE,
    )

    model = LogisticRegression(
        C=C,
        class_weight=CLASS_WEIGHT,
        max_iter=3000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    pseudo_labels = {}

    if candidate_ids:
        candidate_texts = [
            user_turns(dialogs_by_id[dialog_id])
            for dialog_id in candidate_ids
        ]

        X_candidates, _ = vectorize_topic_texts(
            candidate_texts,
            vectorizer=vectorizer,
        )

        probabilities = model.predict_proba(X_candidates)
        best_columns = probabilities.argmax(axis=1)

        predictions = model.classes_[best_columns]
        confidence = probabilities.max(axis=1)

        # Пустые TF-IDF-векторы не добавляем.
        nonempty = np.asarray(
            X_candidates.getnnz(axis=1)
        ).ravel() > 0

        for dialog_id, label, score, has_features in zip(
            candidate_ids,
            predictions,
            confidence,
            nonempty,
        ):
            if has_features and score >= CONFIDENCE_THRESHOLD:
                pseudo_labels[dialog_id] = int(label)

    expanded_train = {
        **train_labels,
        **pseudo_labels,
    }

    save_labels('train_split_ver3.txt', expanded_train)
    save_labels('val_split_ver3.txt', val_labels)
    save_labels('test_split_ver3.txt', test_labels)

    save_labels('pseudo_labels_ver3.txt', pseudo_labels)

    print('Исходный train:', len(train_labels))
    print('Кандидатов после исключения групп:', len(candidate_ids))
    print('Добавлено псевдометок:', len(pseudo_labels))
    print('Расширенный train:', len(expanded_train))
    print('Val без изменений:', len(val_labels))
    print('Test без изменений:', len(test_labels))

    labels, counts = np.unique(
        list(pseudo_labels.values()),
        return_counts=True,
    )

    print('\nДобавлено по классам:')
    for label, count in zip(labels, counts):
        print(f'{label}: {count}')


if __name__ == '__main__':
    main()