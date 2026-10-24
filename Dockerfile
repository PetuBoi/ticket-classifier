FROM python:3.13-slim-bookworm AS base

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --locked --no-dev --no-editable --no-cache

FROM base AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir /artifacts \
    && chown app:app /artifacts
COPY --from=builder /app/.venv /app/.venv
COPY data/tickets.csv data/tickets.csv
USER 10001:10001
ENTRYPOINT ["ticket-classifier"]
CMD ["--help"]
