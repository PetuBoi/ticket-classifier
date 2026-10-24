# IT Ticket Classifier

A small junior IT/DevOps portfolio demo: classify an English IT request as
`account_access`, `hardware`, `network`, or `software`. Python 3.13, uv,
scikit-learn, argparse, joblib, pytest, and Ruff; no external services.

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from
this directory. uv installs Python 3.13 if necessary. `uv.lock` records resolved
dependencies; `--locked` checks that it matches the project.

```console
uv python install 3.13
uv sync --locked
uv run ticket-classifier train --data data/tickets.csv --output artifacts
uv run ticket-classifier predict --model artifacts/model.joblib --text "I cannot connect to the office VPN"
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

If the default uv cache or Python installation directory is not writable, use
these PowerShell settings before the commands above (also used to verify this
project in the restricted development environment):

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.cache\python'
$env:UV_PYTHON_BIN_DIR = Join-Path $PWD '.cache\bin'
```

The CLI exits with status 2 and an error on stderr for invalid arguments,
invalid CSV, a missing model, or blank prediction text. Run
`uv run ticket-classifier --help` for the command summary.

## Docker

Requires Docker with Linux containers. These commands work in PowerShell and
POSIX shells; use a new named volume for the first run:

```console
docker build --tag ticket-classifier:local .
docker volume create ticket-classifier-model
docker run --rm --mount type=volume,source=ticket-classifier-model,target=/artifacts --entrypoint python ticket-classifier:local -c "import os; from pathlib import Path; assert os.getuid() != 0; p = Path('/artifacts/.write-test'); p.write_text('ok'); p.unlink(); print('Non-root volume write OK')"
docker run --rm --mount type=volume,source=ticket-classifier-model,target=/artifacts ticket-classifier:local train --data data/tickets.csv --output /artifacts
docker run --rm --mount type=volume,source=ticket-classifier-model,target=/artifacts,readonly ticket-classifier:local predict --model /artifacts/model.joblib --text "I cannot connect to the office VPN"
```

The image uses Python 3.13 and installs the project and locked runtime dependencies
only (`uv sync --locked --no-dev --no-editable`). uv and build tools stay in the
builder stage. No model is trained during the build. The CLI runs as UID/GID
10001. Only explicitly selected project files enter the image; no credentials
are needed or copied.

`/artifacts` is owned by this user in the image. Docker initializes a fresh empty
named volume from that directory, including ownership. The explicit write check
above verifies permissions before training. An existing volume with different
ownership may need administrator repair; bind mounts use host permissions instead.
See [Docker volume initialization](https://docs.docker.com/engine/storage/volumes/#mounting-a-volume-over-existing-data).

The model and all three reports are stored in the Docker-managed
`ticket-classifier-model` volume, not the local `artifacts/` directory. They survive
`--rm` and are reused by subsequent containers. Prediction only loads the model;
it does not retrain, and its volume is mounted read-only. Always attach this volume
when saving outputs. Removing the named volume deletes its saved model and reports.

## GitHub Actions and verification status

[CI configuration](.github/workflows/ci.yml) runs on `push` and `pull_request`:
locked installation with Python 3.13 and uv, Ruff lint/format checks, pytest,
training and prediction smoke checks, then Docker build, non-root volume write
check, training, and prediction in a new container sharing the saved model.
There is no accuracy threshold, deployment, image publication, or secret required.
Token permissions are limited to `contents: read`, and checkout does not persist
credentials. Actions use supported versions:
[checkout v7](https://github.com/actions/checkout/releases),
[setup-python v7](https://github.com/actions/setup-python/releases), and
[setup-uv v10](https://github.com/astral-sh/setup-uv/releases).

**Locally verified:** locked installation, Ruff lint and formatting, all 37 pytest
tests, and native CLI training/prediction. Model scores are recorded below.
**Docker verified locally on 2026-10-23:** image build on Docker Desktop with Linux
containers (Python 3.13.17), UID/GID `10001:10001`, and successful write/delete in a
fresh named volume. Container training saved all four artifacts and reproduced
accuracy **0.8542** and macro F1 **0.8515**. After the training container was
automatically removed, a new container loaded the saved model from the same volume
mounted read-only. The VPN example returned `network` (estimated probability
**0.4871**, not calibrated). Saved scenario lists do not overlap; pytest and Ruff
are absent from the runtime image. No Dockerfile changes were needed.

Verification used image `ticket-classifier:verify-20261023` and named volume
`ticket-classifier-verify-20261023`; both remain available locally. All verification
containers used `--rm` and were removed. No unrelated Docker resources were changed.

**Not yet verified:** the GitHub Actions workflow has not run on GitHub; its first
hosted run is still required. Nothing has been pushed or published.

## Data and evaluation

`data/tickets.csv` contains **192 synthetic requests**, 48 per category:
16 scenarios per category, each with three paraphrases. Columns are `text`,
`category`, and `scenario_id`. Scenario IDs group a concrete issue and its
paraphrases. Account permissions belong to `account_access`; physical device
faults to `hardware`; connectivity faults to `network`; application, driver,
installation, and license issues to `software`. These boundaries are simplified.

The split uses seed **42** and holds out **25% of scenarios**, stratified by
category. Both subsets contain all four categories, and no scenario appears in
both. For the supplied equal-size groups this gives 144 training and 48 test
requests. Sorting scenario IDs makes their assignment independent of CSV order.
TF-IDF vocabulary and IDF weights, and logistic regression parameters, are fitted
only on training requests. Fixed settings use word unigrams/bigrams and logistic
regression with `max_iter=1000`; the test set is used only for final evaluation,
never for tuning. There is no parameter search or score-based data revision.

Training prints accuracy, macro F1, and per-category precision/recall/F1/support
for the model and a `DummyClassifier(strategy="most_frequent")` baseline. The
baseline always predicts the most common training class (ties follow sklearn's
class order). The output directory contains:

- `model.joblib`: the complete fitted TF-IDF + classifier pipeline.
- `metrics.json`: model and baseline metrics, split settings, and sample counts.
- `confusion_matrix.csv`: model counts, rows = actual and columns = predicted.
- `scenario_split.json`: exact train/test scenario IDs and scenario categories.

Verified on Python 3.13.16 with the included lockfile: model accuracy **0.8542**,
macro F1 **0.8515**; baseline accuracy **0.2500**, macro F1 **0.1000**, on 48
synthetic test requests. Ruff and all 37 tests passed. Example CLI predictions
returned `network` for the office VPN request, `account_access` for a password
reset, `hardware` for broken keyboard keys, and `software` for a spreadsheet
crash. These examples are smoke checks, not additional evaluation data.

CSV validation rejects missing/extra/duplicate columns, malformed quoted records,
incorrect field counts, invalid UTF-8, NUL characters, empty texts or scenario IDs,
unknown categories, duplicate texts after trimming surrounding whitespace, and
scenarios with conflicting labels. All categories must be present. Provide at
least four scenarios per category for a straightforward 75/25 split. The loader
accepts UTF-8 BOM, quoted commas/newlines, and a different column order.

## Limitations and trust

All data are synthetic and small. **Reported scores do not represent accuracy
on real company tickets.** Scenario separation reduces direct paraphrase leakage,
but related vocabulary and synthetic writing style still make evaluation limited.
Categories overlap: a printer problem may be a device, driver, permission, or
network issue. This classifier cannot ask follow-up questions or detect an unknown
topic; it forces every input, even unrelated text, into one of the four classes.
Printed class probabilities are model estimates, **not calibrated confidence or
certainty**. No calibration or real-world validation has been performed.

**Load joblib models only from a trusted source.** Deserialization can execute
arbitrary code; checking the resulting object does not make an untrusted file
safe. Model artifacts are local generated files excluded from version control.
Keep this source dataset unchanged when reproducing the saved scenario split;
use the locked dependencies and Python 3.13.

AI (OpenAI Codex) assisted with the implementation, tests, documentation, and
creation of all synthetic examples. The dataset is not collected from users or
company systems. Tests use small local fixtures, make no network requests, and
do not require a high accuracy score. The actual model still needs review and
evaluation on appropriately handled real data before practical use.

Source: `src/ticket_classifier/{data,model,cli}.py`. A short Czech explanation is
in [docs/explanation.md](docs/explanation.md).
