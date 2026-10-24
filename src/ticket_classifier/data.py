"""CSV validation and a stratified split of whole scenarios."""

import csv
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

CATEGORIES = ("account_access", "hardware", "network", "software")
SEED = 42
TEST_SIZE = 0.25


@dataclass(frozen=True)
class Ticket:
    text: str
    category: str
    scenario_id: str


def load_tickets(path: Path) -> list[Ticket]:
    """Reject malformed records instead of silently skipping or repairing them."""
    tickets = []
    seen_texts = set()
    scenario_categories = {}
    try:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            fields = reader.fieldnames
            if (
                fields is None
                or len(fields) != 3
                or set(fields) != {"text", "category", "scenario_id"}
            ):
                raise ValueError(
                    "CSV must have exactly these columns: text, category, scenario_id."
                )
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Invalid CSV field count at line {reader.line_num}.")
                if any("\x00" in value for value in row.values()):
                    raise ValueError(f"NUL character at line {reader.line_num}.")
                text, category, scenario = (
                    row[key].strip() for key in ("text", "category", "scenario_id")
                )
                if not text:
                    raise ValueError(f"Empty text at line {reader.line_num}.")
                if category not in CATEGORIES:
                    raise ValueError(f"Unknown category at line {reader.line_num}: {category!r}.")
                if not scenario:
                    raise ValueError(f"Empty scenario_id at line {reader.line_num}.")
                if text in seen_texts:
                    raise ValueError(f"Duplicate text at line {reader.line_num}: {text!r}.")
                if scenario in scenario_categories and scenario_categories[scenario] != category:
                    raise ValueError(f"Scenario {scenario!r} has multiple categories.")
                seen_texts.add(text)
                scenario_categories[scenario] = category
                tickets.append(Ticket(text, category, scenario))
    except (csv.Error, UnicodeError) as error:
        raise ValueError(f"Invalid CSV: {error}") from error
    if not tickets:
        raise ValueError("CSV contains no tickets.")
    if {ticket.category for ticket in tickets} != set(CATEGORIES):
        raise ValueError("Dataset must contain all four categories.")
    return tickets


def split_tickets(tickets: list[Ticket]) -> tuple[list[Ticket], list[Ticket]]:
    """Stratify scenario IDs by category, then expand them back into tickets."""
    scenarios = {}
    for ticket in tickets:
        if ticket.scenario_id in scenarios and scenarios[ticket.scenario_id] != ticket.category:
            raise ValueError(f"Scenario {ticket.scenario_id!r} has multiple categories.")
        scenarios[ticket.scenario_id] = ticket.category
    ids = sorted(scenarios)
    if set(scenarios.values()) != set(CATEGORIES):
        raise ValueError("Dataset must contain all four categories.")
    try:
        train_ids, test_ids = train_test_split(
            ids,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=[scenarios[scenario] for scenario in ids],
        )
    except ValueError as error:
        raise ValueError(
            "Not enough scenarios for a stratified 75/25 split; provide at least four "
            "scenarios per category."
        ) from error
    train_ids, test_ids = set(train_ids), set(test_ids)
    train = [ticket for ticket in tickets if ticket.scenario_id in train_ids]
    test = [ticket for ticket in tickets if ticket.scenario_id in test_ids]
    if any({ticket.category for ticket in subset} != set(CATEGORIES) for subset in (train, test)):
        raise ValueError("Both split subsets must contain all four categories.")
    return train, test
