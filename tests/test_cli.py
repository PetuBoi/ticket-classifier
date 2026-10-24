import subprocess

import pytest

from ticket_classifier.cli import main


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "required"),
        (["invalid"], "invalid choice"),
        (["train"], "required"),
        (["predict", "--model", "missing.joblib"], "required"),
        (["predict", "--model", "missing.joblib", "--text", "help"], "Model file not found"),
        (["predict", "--model", "missing.joblib", "--text", "   "], "must not be empty"),
    ],
)
def test_invalid_cli(args, message, capsys):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_train_and_predict_cli(data_path, tmp_path, capsys):
    output = tmp_path / "cli-artifacts"
    assert main(["train", "--data", str(data_path), "--output", str(output)]) == 0
    text = capsys.readouterr().out
    assert "accuracy=" in text
    assert "macro F1=" in text
    assert "baseline:" in text
    assert "precision" in text and "recall" in text
    assert (
        main(
            [
                "predict",
                "--model",
                str(output / "model.joblib"),
                "--text",
                "The application crashes",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "Category:" in text and "not calibrated" in text


def test_invalid_training_csv_cli(tmp_path, capsys):
    data = tmp_path / "invalid.csv"
    data.write_text("wrong,columns\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["train", "--data", str(data), "--output", str(tmp_path / "output")])
    assert error.value.code == 2
    assert "CSV must have exactly" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_missing_training_csv_cli(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["train", "--data", str(tmp_path / "missing.csv"), "--output", str(tmp_path)])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_installed_entrypoint(tmp_path):
    result = subprocess.run(
        [
            "ticket-classifier",
            "predict",
            "--model",
            str(tmp_path / "missing.joblib"),
            "--text",
            "help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Model file not found" in result.stderr
    assert "Traceback" not in result.stderr
