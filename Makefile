POETRY_PYTHON ?= python3
SRC_DIRS = src scripts
MYPY_DIRS = src/common/data

-include .env
export

init_venv:
	poetry env use ${POETRY_PYTHON}

install_poetry:  # Only for CI
	pip install poetry==2.1.1

install_dev: init_venv
	poetry install --with dev

lint: install_dev
	poetry check --lock
	flake8 ${SRC_DIRS}
	black ${SRC_DIRS} --check
	isort ${SRC_DIRS} --check-only --diff
	autoflake ${SRC_DIRS} --check
	# for now we only check data abstractions and services packages
	mypy --show-error-codes ${MYPY_DIRS}

test_db_migrate: export PGVECTOR_HOST=$(TEST_DATABASE_HOST)
test_db_migrate: export PGVECTOR_PORT=$(TEST_DATABASE_PORT)
test_db_migrate: export PGVECTOR_DATABASE=$(TEST_DATABASE)
test_db_migrate: export ELASTIC_CONNECTION_STRING=$(TEST_ELASTIC_CONNECTION_STRING)
test_db_migrate:
	alembic -c src/alembic.ini upgrade head

test_unit: export PGVECTOR_HOST=$(TEST_DATABASE_HOST)
test_unit: export PGVECTOR_PORT=$(TEST_DATABASE_PORT)
test_unit: export PGVECTOR_DATABASE=$(TEST_DATABASE)
test_unit: export ELASTIC_CONNECTION_STRING=$(TEST_ELASTIC_CONNECTION_STRING)
test_unit:
	pytest tests/unit --junitxml=reports/tests-unit.xml

test_integration: export EMBEDDING_DEFAULT_MODEL=text-embedding-3-large
test_integration: export PGVECTOR_HOST=$(TEST_DATABASE_HOST)
test_integration: export PGVECTOR_PORT=$(TEST_DATABASE_PORT)
test_integration: export PGVECTOR_DATABASE=$(TEST_DATABASE)
test_integration: export ELASTIC_CONNECTION_STRING=$(TEST_ELASTIC_CONNECTION_STRING)
test_integration: test_db_migrate
	pytest tests/integration --junitxml=reports/tests-int.xml

test: test_unit test_integration
