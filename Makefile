POETRY_PYTHON ?= $(if $(pythonLocation),$(pythonLocation)/bin/python,python3)
SRC_DIRS = src scripts tests
MYPY_DIRS = src/common src/admin_portal src/statgpt

-include .env
export

# AI DIAL SDK: pydantic v2 mode
export PYDANTIC_V2=True

remove_venv:
	poetry env remove --all || true
	$(POETRY_PYTHON) -m venv .venv

init_venv:
	poetry env use .venv/bin/python

install_dev: init_venv
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
# Check if GNU gettext tools are installed
check_gettext:
ifeq ($(OS),Windows_NT)
	@where xgettext >nul 2>&1 || ( \
		echo Error: xgettext not found. GNU gettext tools are required for localization. & \
		echo. & \
		echo Installation instructions: & \
		echo   MacOS:       brew install gettext & \
		echo   Linux/WSL:   sudo apt install gettext & \
		echo   Windows:     choco install gettext & \
		echo. & \
		echo See README.md for more details. & \
		exit /b 1 \
	)
	@where msgmerge >nul 2>&1 || (echo Error: msgmerge not found. Please install GNU gettext tools. & exit /b 1)
	@where msgfmt >nul 2>&1 || (echo Error: msgfmt not found. Please install GNU gettext tools. & exit /b 1)
else
	@command -v xgettext >/dev/null 2>&1 || { \
		echo "Error: xgettext not found. GNU gettext tools are required for localization."; \
		echo ""; \
		echo "Installation instructions:"; \
		echo "  MacOS:       brew install gettext"; \
		echo "  Linux/WSL:   sudo apt install gettext"; \
		echo "  Windows:     choco install gettext"; \
		echo ""; \
		echo "See README.md for more details."; \
		exit 1; \
	}
	@command -v msgmerge >/dev/null 2>&1 || { \
		echo "Error: msgmerge not found. Please install GNU gettext tools."; \
		exit 1; \
	}
	@command -v msgfmt >/dev/null 2>&1 || { \
		echo "Error: msgfmt not found. Please install GNU gettext tools."; \
		exit 1; \
	}
endif

extract_messages: check_gettext
	@echo "Extracting translatable strings from formatters..."
	@cd src/statgpt/utils/formatters && \
	xgettext -d dataset -o locales/dataset.pot \
		--language=Python \
		--keyword=_ \
		--from-code=UTF-8 \
		base.py dataset_base.py dataset_simple.py dataset_detailed.py datasets_list_formatter.py citation.py \
		dataset_query.py dataset_availablity_query.py

update_messages: check_gettext
	@echo "Updating .po files from template..."
	@cd src/statgpt/utils/formatters/locales && \
	msgmerge --update en/LC_MESSAGES/dataset.po dataset.pot && \
	msgmerge --update uk/LC_MESSAGES/dataset.po dataset.pot

compile_messages: check_gettext
	@echo "Compiling .po files to .mo files..."
	@cd src/statgpt/utils/formatters/locales && \
	msgfmt -o en/LC_MESSAGES/dataset.mo en/LC_MESSAGES/dataset.po && \
	msgfmt -o uk/LC_MESSAGES/dataset.mo uk/LC_MESSAGES/dataset.po

# Convenience command to compile messages after changes
locales: compile_messages
