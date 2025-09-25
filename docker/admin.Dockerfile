FROM python:3.11-slim

ARG GIT_COMMIT
ENV GIT_COMMIT=$GIT_COMMIT

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED=1
ENV PIP_ARGS="--progress-bar off --no-cache-dir"

ENV APP_HOME=/home/app

WORKDIR $APP_HOME

# Install dependencies using poetry
RUN pip install $PIP_ARGS "poetry==2.1.1"
RUN poetry self add poetry-plugin-export
COPY pyproject.toml .
COPY poetry.lock .
RUN poetry export -f requirements.txt --without-hashes | pip install $PIP_ARGS -r /dev/stdin

# Copy scripts and source code
COPY ./src/alembic.ini $APP_HOME/alembic.ini
COPY ./scripts/admin.sh $APP_HOME/admin.sh
COPY ./src/admin_portal $APP_HOME/admin_portal
COPY ./src/common $APP_HOME/common

# create the app user and chown workdir to the app user
RUN adduser -u 5678 --system --disabled-password --gecos "" app && chown -R app $APP_HOME
USER app

ENV APP_MODE="DIAL"
ENV WEB_CONCURRENCY=1

CMD ["sh", "admin.sh"]
