import csv

import pytest

from ticket_classifier.data import CATEGORIES, Ticket, load_tickets, split_tickets


def test_load_and_split(data_path, tickets):
    loaded = load_tickets(data_path)
    assert loaded == tickets
    train, test = split_tickets(loaded)
    assert not {row.scenario_id for row in train} & {row.scenario_id for row in test}
    assert set(train) | set(test) == set(tickets)
    assert len(train) + len(test) == len(tickets)
    assert all({row.category for row in part} == set(CATEGORIES) for part in (train, test))
    assert split_tickets(loaded) == (train, test)
    assert split_tickets(list(reversed(loaded))) == (list(reversed(train)), list(reversed(test)))


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "columns"),
        ("text,category\nhello,hardware\n", "columns"),
        ("text,category,category\nhello,hardware,hardware\n", "columns"),
        ("text,category,scenario_id,extra\n", "columns"),
        ("text,category,scenario_id\n", "no tickets"),
        ("text,category,scenario_id\n,hardware,h1\n", "Empty text"),
        ("text,category,scenario_id\n   ,hardware,h1\n", "Empty text"),
        ("text,category,scenario_id\nhello,unknown,h1\n", "Unknown category"),
        ("text,category,scenario_id\nhello,hardware,\n", "Empty scenario_id"),
        ("text,category,scenario_id\nhello,hardware,h1,extra\n", "field count"),
        ("text,category,scenario_id\nhello,hardware\n", "field count"),
        ('text,category,scenario_id\n"unterminated,hardware,h1\n', "Invalid CSV"),
        ("text,category,scenario_id\nhello\x00,hardware,h1\n", "NUL"),
        ("text,category,scenario_id\nhello,hardware,h1\nhello,hardware,h2\n", "Duplicate text"),
        ("text,category,scenario_id\nhello,hardware,h1\n hello ,hardware,h2\n", "Duplicate text"),
        ("text,category,scenario_id\nhello,hardware,h1\nworld,network,h1\n", "multiple categories"),
        ("text,category,scenario_id\nhello,hardware,h1\n", "all four categories"),
    ],
)
def test_invalid_csv(tmp_path, content, message):
    path = tmp_path / "invalid.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_tickets(path)


def test_invalid_encoding(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"text,category,scenario_id\n\xff,hardware,h1\n")
    with pytest.raises(ValueError, match="Invalid CSV"):
        load_tickets(path)


def test_csv_quoting_and_bom(data_path, tickets):
    with data_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["scenario_id", "category", "text"])
        writer.writerows(
            (row.scenario_id, row.category, row.text + ', says "help"\nnow') for row in tickets
        )
    assert len(load_tickets(data_path)) == len(tickets)


def test_insufficient_scenarios(tickets):
    with pytest.raises(ValueError, match="Not enough scenarios"):
        split_tickets([row for row in tickets if row.scenario_id.endswith("_0")])


def test_split_rejects_mixed_scenario(tickets):
    with pytest.raises(ValueError, match="multiple categories"):
        split_tickets([*tickets, Ticket("other", "network", tickets[0].scenario_id)])
