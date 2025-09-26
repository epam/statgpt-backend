POETRY_PYTHON ?= $(if $(pythonLocation),$(pythonLocation)/bin/python,python3)
SRC_DIRS = src scripts
STATGPT_MYPY_DIRS = src/statgpt/config src/statgpt/default_prompts src/statgpt/schemas src/statgpt/security src/statgpt/services src/statgpt/settings src/statgpt/utils/formatters src/statgpt/utils/openai src/statgpt/utils/message_interceptors src/statgpt/utils/message_history.py
MYPY_DIRS = src/common src/admin_portal ${STATGPT_MYPY_DIRS}

-include .env
export

init_venv:
	@echo "Using Python: $(POETRY_PYTHON)"
	@$(POETRY_PYTHON) --version
	rm -rf .venv || true
	poetry env remove --all || true
	$(POETRY_PYTHON) -m venv .venv
	poetry env use .venv/bin/python
	@echo "Verifying virtual environment:"
	.venv/bin/python --version
	@echo "Poetry environment info:"
	poetry env info

install_dev: init_venv
	@echo "About to run poetry install..."
	poetry install --with dev

format: install_dev
	autoflake ${SRC_DIRS}
	black ${SRC_DIRS}
	isort ${SRC_DIRS}

lint: install_dev
	poetry check --lock
	poetry run flake8 ${SRC_DIRS}
	poetry run black ${SRC_DIRS} --check
	poetry run isort ${SRC_DIRS} --check-only --diff
	poetry run autoflake ${SRC_DIRS} --check
	# for now we only check data abstractions and services packages
	poetry run mypy --show-error-codes ${MYPY_DIRS}

install_pre_commit_hooks:
	pre-commit install

db_migrate:
	alembic -c src/alembic.ini upgrade head

db_downgrade:
	alembic -c src/alembic.ini downgrade -1

db_autogenerate:
	alembic -c src/alembic.ini revision --autogenerate -m "$(MESSAGE)"

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

# Localization commands for dataset formatters
extract_messages:
	@echo "Extracting translatable strings from formatters..."
	@cd src/statgpt/utils/formatters && \
	xgettext -d dataset -o locales/dataset.pot \
		--language=Python \
		--keyword=_ \
		--from-code=UTF-8 \
		base.py dataset_base.py dataset_simple.py dataset_detailed.py datasets_list_formatter.py citation.py

update_messages:
	@echo "Updating .po files from template..."
	@cd src/statgpt/utils/formatters/locales && \
	msgmerge --update en/LC_MESSAGES/dataset.po dataset.pot && \
	msgmerge --update uk/LC_MESSAGES/dataset.po dataset.pot

compile_messages:
	@echo "Compiling .po files to .mo files..."
	@cd src/statgpt/utils/formatters/locales && \
	msgfmt -o en/LC_MESSAGES/dataset.mo en/LC_MESSAGES/dataset.po && \
	msgfmt -o uk/LC_MESSAGES/dataset.mo uk/LC_MESSAGES/dataset.po

# Convenience command to compile messages after changes
locales: compile_messages
