import csv
import json

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ticket_classifier.data import CATEGORIES, SEED, load_tickets, split_tickets
from ticket_classifier.model import predict_ticket, train_model


def test_train_predict_and_persist(data_path, tmp_path):
    output = tmp_path / "artifacts"
    metrics = train_model(data_path, output)
    pipeline = joblib.load(output / "model.joblib")
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
    assert json.loads((output / "metrics.json").read_text()) == metrics
    for name in ("model", "baseline"):
        assert 0 <= metrics[name]["accuracy"] <= 1
        assert 0 <= metrics[name]["macro_f1"] <= 1
        assert set(CATEGORIES) <= metrics[name]["classification_report"].keys()
    train, test = split_tickets(load_tickets(data_path))
    assert metrics["train_samples"] == len(train)
    assert metrics["test_samples"] == len(test)
    vocabulary = pipeline.named_steps["tfidf"].vocabulary_
    # Each fixture scenario has a unique marker: held-out markers must never be fitted.
    for ticket in test:
        marker = next(word for word in ticket.text.split() if word.startswith("marker"))
        assert marker not in vocabulary
    for ticket in train:
        marker = next(word for word in ticket.text.split() if word.startswith("marker"))
        assert marker in vocabulary
    split = json.loads((output / "scenario_split.json").read_text())
    assert split["seed"] == SEED
    assert set(split["train_scenarios"]) == {row.scenario_id for row in train}
    assert set(split["test_scenarios"]) == {row.scenario_id for row in test}
    assert not set(split["train_scenarios"]) & set(split["test_scenarios"])
    with (output / "confusion_matrix.csv").open(newline="") as source:
        matrix = list(csv.reader(source))
    assert matrix[0] == ["actual/predicted", *CATEGORIES]
    assert [row[0] for row in matrix[1:]] == list(CATEGORIES)
    assert sum(int(value) for row in matrix[1:] for value in row[1:]) == len(test)
    category, probabilities = predict_ticket(output / "model.joblib", "My keyboard is broken")
    assert category in CATEGORIES
    assert set(probabilities) == set(CATEGORIES)
    assert all(0 <= value <= 1 for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert category == max(probabilities, key=probabilities.get)
    assert category == pipeline.predict(["My keyboard is broken"])[0]
    assert list(probabilities.values()) == pytest.approx(
        pipeline.predict_proba(["My keyboard is broken"])[0]
    )
    again = train_model(data_path, tmp_path / "again")
    assert again == metrics


def test_missing_model(tmp_path):
    with pytest.raises(ValueError, match="Model file not found"):
        predict_ticket(tmp_path / "missing.joblib", "Need help")


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_empty_prediction(tmp_path, text):
    with pytest.raises(ValueError, match="must not be empty"):
        predict_ticket(tmp_path / "missing.joblib", text)


def test_wrong_model(tmp_path):
    path = tmp_path / "wrong.joblib"
    joblib.dump({"not": "a classifier"}, path)
    with pytest.raises(ValueError, match="trained ticket-classifier pipeline"):
        predict_ticket(path, "Need help")
