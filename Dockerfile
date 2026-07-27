FROM python:3.11-slim

WORKDIR /workspace

COPY pyproject.toml README.md Makefile ./
COPY src ./src
COPY tests ./tests
COPY protocol ./protocol

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[live,test]"

CMD ["python", "-m", "pytest", "-q"]
