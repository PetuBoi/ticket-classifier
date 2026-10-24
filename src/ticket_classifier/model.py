"""Train once on the training split and report held-out metrics."""

import csv
import json
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from ticket_classifier.data import CATEGORIES, SEED, TEST_SIZE, load_tickets, split_tickets


def evaluate(expected: list[str], predicted) -> dict:
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, labels=CATEGORIES, average="macro")),
        "classification_report": classification_report(
            expected,
            predicted,
            labels=CATEGORIES,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_model(data_path: Path, output: Path) -> dict:
    tickets = load_tickets(data_path)
    train, test = split_tickets(tickets)
    train_texts = [ticket.text for ticket in train]
    train_labels = [ticket.category for ticket in train]
    test_texts = [ticket.text for ticket in test]
    test_labels = [ticket.category for ticket in test]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    predicted = pipeline.predict(test_texts)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit([[0]] * len(train), train_labels)
    metrics = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "train_samples": len(train),
        "test_samples": len(test),
        "categories": list(CATEGORIES),
        "model": evaluate(test_labels, predicted),
        "baseline": evaluate(test_labels, baseline.predict([[0]] * len(test))),
        "baseline_strategy": "most_frequent",
        "data_note": "Synthetic data only; these scores do not measure real-world accuracy.",
    }
    split = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "train_scenarios": sorted({ticket.scenario_id for ticket in train}),
        "test_scenarios": sorted({ticket.scenario_id for ticket in test}),
        "scenario_categories": dict(
            sorted({ticket.scenario_id: ticket.category for ticket in tickets}.items())
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output / "model.joblib")
    for filename, value in (("metrics.json", metrics), ("scenario_split.json", split)):
        (output / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["actual/predicted", *CATEGORIES])
        matrix = confusion_matrix(test_labels, predicted, labels=CATEGORIES)
        for category, row in zip(CATEGORIES, matrix, strict=True):
            writer.writerow([category, *row])
    return metrics


def predict_ticket(model_path: Path, text: str) -> tuple[str, dict[str, float]]:
    if not text.strip():
        raise ValueError("Ticket text must not be empty.")
    if not model_path.is_file():
        raise ValueError(f"Model file not found: {model_path}. Train a model first.")
    # joblib uses pickle: callers must only supply trusted model files.
    pipeline = joblib.load(model_path)
    if (
        not isinstance(pipeline, Pipeline)
        or list(pipeline.named_steps) != ["tfidf", "classifier"]
        or not isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
        or not isinstance(pipeline.named_steps["classifier"], LogisticRegression)
        or set(pipeline.classes_) != set(CATEGORIES)
    ):
        raise ValueError("Model must be a trained ticket-classifier pipeline.")
    category = str(pipeline.predict([text.strip()])[0])
    probabilities = {
        str(label): float(probability)
        for label, probability in zip(
            pipeline.classes_,
            pipeline.predict_proba([text.strip()])[0],
            strict=True,
        )
    }
    return category, probabilities
