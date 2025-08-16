POETRY_PYTHON ?= python3
SRC_DIRS = src scripts
MYPY_DIRS = src/common/data

-include .env
export

init_venv:
	poetry env use ${POETRY_PYTHON}

install_dev: init_venv
	poetry install --with dev

lint: install_dev
	poetry check --lock
	poetry run flake8 ${SRC_DIRS}
	poetry run black ${SRC_DIRS} --check
	poetry run isort ${SRC_DIRS} --check-only --diff
	poetry run autoflake ${SRC_DIRS} --check
	# for now we only check data abstractions and services packages
	poetry run mypy --show-error-codes ${MYPY_DIRS}

test_db_migrate: export PGVECTOR_HOST=$(TEST_DATABASE_HOST)
test_db_migrate: export PGVECTOR_PORT=$(TEST_DATABASE_PORT)
test_db_migrate: export PGVECTOR_DATABASE=$(TEST_DATABASE)
test_db_migrate: export ELASTIC_CONNECTION_STRING=$(TEST_ELASTIC_CONNECTION_STRING)
test_db_migrate: install_dev
	poetry run alembic -c src/alembic.ini upgrade head

test_unit: export PGVECTOR_HOST=$(TEST_DATABASE_HOST)
test_unit: export PGVECTOR_PORT=$(TEST_DATABASE_PORT)
test_unit: export PGVECTOR_DATABASE=$(TEST_DATABASE)
test_unit: export ELASTIC_CONNECTION_STRING=$(TEST_ELASTIC_CONNECTION_STRING)
test_unit: install_dev
	poetry run pytest tests/unit --junitxml=reports/tests-unit.xml

test_integration: export EMBEDDING_DEFAULT_MODEL=text-embedding-3-large
test_integration: export PGVECTOR_HOST=$(TEST_DATABASE_HOST)
test_integration: export PGVECTOR_PORT=$(TEST_DATABASE_PORT)
test_integration: export PGVECTOR_DATABASE=$(TEST_DATABASE)
test_integration: export ELASTIC_CONNECTION_STRING=$(TEST_ELASTIC_CONNECTION_STRING)
test_integration: test_db_migrate
	poetry run pytest tests/integration --junitxml=reports/tests-int.xml

test: test_unit test_integration
