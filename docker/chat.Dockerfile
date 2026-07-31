############################
# Builder stage
############################
FROM python:3.11-alpine AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off \
    PIP_ROOT_USER_ACTION=ignore \
    APP_HOME=/home/app

RUN pip install --upgrade pip && pip install poetry==2.2.1

WORKDIR $APP_HOME

# Install dependencies first to leverage Docker layer caching
COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --no-interaction --no-ansi --no-cache --no-root \
  --no-directory --only main -E mcp

# Copy source code and install the project
COPY ./statgpt/app $APP_HOME/statgpt/app
COPY ./statgpt/common $APP_HOME/statgpt/common
RUN poetry install --no-interaction --no-ansi --no-cache --no-root --only main -E mcp

# CVE-2026-23949 (jaraco.context vendored in setuptools), CVE-2026-24049 (wheel vendored in setuptools)
RUN .venv/bin/pip install "setuptools==80.10.2" "wheel==0.46.2"

############################
# Runtime stage
############################
FROM python:3.11-alpine AS server

# Security patches (consolidated into single layer)
# CVE-2023-52425 (libexpat), CVE-2025-6965 (sqlite-libs), libcrypto3/libssl3
# CVE-2026-40200 (musl)
RUN apk update && apk upgrade --no-cache \
    libcrypto3 libssl3 libexpat sqlite-libs zlib musl musl-utils \
  && apk add --no-cache ca-certificates \
  && update-ca-certificates \
  && rm -rf /var/cache/apk/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    APP_HOME=/home/app \
    WEB_CONCURRENCY=1 \
    PYDANTIC_V2=True

# CVE-2026-23949 (jaraco.context vendored in setuptools), CVE-2026-24049 (wheel vendored in setuptools)
RUN pip install --upgrade pip \
  && pip install "setuptools==80.10.2" "wheel==0.46.2"

WORKDIR $APP_HOME

# Create non-root user and copy built application
RUN adduser -u 1001 --disabled-password --gecos "" appuser
COPY --chown=appuser --from=builder $APP_HOME .

# The service runs from the prebuilt venv and never invokes pip/venv at runtime.
# Remove packaging tooling so its vendored/bundled deps (pip's msgpack,
# ensurepip's setuptools wheel) aren't flagged as CVEs. The installed
# setuptools/wheel stay pinned & patched.
RUN rm -rf /usr/local/lib/python3.11/ensurepip \
 && rm -rf /usr/local/lib/python3.11/site-packages/pip \
           /usr/local/lib/python3.11/site-packages/pip-*.dist-info \
           /usr/local/bin/pip* \
 && rm -rf /home/app/.venv/lib/python3.11/site-packages/pip \
           /home/app/.venv/lib/python3.11/site-packages/pip-*.dist-info \
           /home/app/.venv/bin/pip*

COPY --chmod=755 ./docker/scripts/chat_docker_entrypoint.sh /docker_entrypoint.sh

EXPOSE 5000

USER appuser

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
  CMD wget -q --spider -T 3 http://localhost:5000/health || exit 1

ARG GIT_COMMIT
ENV GIT_COMMIT=$GIT_COMMIT

ENTRYPOINT ["/docker_entrypoint.sh"]
