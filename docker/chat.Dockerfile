############################
# Builder stage
############################
FROM python:3.11-alpine AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/home/app

RUN pip install --progress-bar off --no-cache-dir poetry==2.2.1

WORKDIR $APP_HOME

# Install dependencies first to leverage Docker layer caching
COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --no-interaction --no-ansi --no-cache --no-root \
  --no-directory --only main

# Copy source code and install the project
COPY ./statgpt/app $APP_HOME/statgpt/app
COPY ./statgpt/common $APP_HOME/statgpt/common
RUN poetry install --no-interaction --no-ansi --no-cache --no-root --only main

############################
# Runtime stage
############################
FROM python:3.11-alpine AS server

# Security patches (consolidated into single layer)
# CVE-2023-52425 (libexpat), CVE-2025-6965 (sqlite-libs), libcrypto3/libssl3
RUN apk update && apk upgrade --no-cache \
    libcrypto3 libssl3 libexpat sqlite-libs \
  && apk add --no-cache ca-certificates \
  && update-ca-certificates \
  && rm -rf /var/cache/apk/*

# CVE-2026-23949 (setuptools), CVE-2026-24049 (wheel)
RUN pip install --no-cache-dir setuptools==80.10.2 wheel==0.46.2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/home/app \
    WEB_CONCURRENCY=1 \
    PYDANTIC_V2=True

WORKDIR $APP_HOME

# Create non-root user and copy built application
RUN adduser -u 1001 --disabled-password --gecos "" appuser
COPY --chown=appuser --from=builder $APP_HOME .

COPY --chmod=755 ./docker/scripts/chat_docker_entrypoint.sh /docker_entrypoint.sh

EXPOSE 5000

USER appuser

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health || exit 1

ARG GIT_COMMIT
ENV GIT_COMMIT=$GIT_COMMIT

ENTRYPOINT ["/docker_entrypoint.sh"]
