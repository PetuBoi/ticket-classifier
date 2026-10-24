import csv

import pytest

from ticket_classifier.data import CATEGORIES, Ticket


@pytest.fixture
def tickets():
    topics = [
        "password account login",
        "broken keyboard screen",
        "wifi router connection",
        "application install crash",
    ]
    return [
        Ticket(
            f"{topic} marker{index}scenario{scenario} request {variant}",
            category,
            f"{category}_{scenario}",
        )
        for index, (category, topic) in enumerate(zip(CATEGORIES, topics, strict=True))
        for scenario in range(4)
        for variant in ("please help", "needs attention")
    ]


@pytest.fixture
def data_path(tmp_path, tickets):
    path = tmp_path / "tickets.csv"
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["text", "category", "scenario_id"])
        writer.writerows((ticket.text, ticket.category, ticket.scenario_id) for ticket in tickets)
    return path
