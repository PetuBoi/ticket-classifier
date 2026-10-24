"""Command line interface for training and prediction."""

import argparse
import pickle
from pathlib import Path

from ticket_classifier.data import CATEGORIES
from ticket_classifier.model import predict_ticket, train_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify English IT support tickets.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train and evaluate on synthetic ticket data.")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    predict = commands.add_parser("predict", help="Predict with a trusted saved model.")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--text", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            metrics = train_model(args.data, args.output)
            print("Synthetic held-out data; scores do not represent real-world accuracy.")
            print(f"Split: {metrics['train_samples']} train / {metrics['test_samples']} test")
            for name in ("model", "baseline"):
                result = metrics[name]
                print(
                    f"\n{name}: accuracy={result['accuracy']:.4f} macro F1={result['macro_f1']:.4f}"
                )
                print(f"{'category':<18} {'precision':>9} {'recall':>9} {'F1':>9} {'support':>8}")
                for category in CATEGORIES:
                    row = result["classification_report"][category]
                    print(
                        f"{category:<18} {row['precision']:9.4f} {row['recall']:9.4f} "
                        f"{row['f1-score']:9.4f} {row['support']:8.0f}"
                    )
            print(f"\nArtifacts saved to {args.output}")
        else:
            category, probabilities = predict_ticket(args.model, args.text)
            print(f"Category: {category}")
            print("Estimated class probabilities (not calibrated):")
            for label, probability in probabilities.items():
                print(f"  {label}: {probability:.4f}")
    except (OSError, ValueError, EOFError, pickle.UnpicklingError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0
