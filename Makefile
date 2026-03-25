SRC_DIRS = statgpt scripts tests
MYPY_DIRS = statgpt scripts
POETRY ?= poetry
PYTHON ?= python3

-include .env
export

# AI DIAL SDK: pydantic v2 mode
export PYDANTIC_V2=True

init_venv:
	$(POETRY) env use $(PYTHON)

install: init_venv
	$(POETRY) install -E cli -E beta-mcp

install_dev: init_venv
	$(POETRY) install -E cli -E beta-mcp --with dev

install_all: init_venv
	$(POETRY) install -E cli -E beta-mcp --with dev,experiments

clean:
	-$(POETRY) env remove --all

format: install_dev
	$(POETRY) run autoflake $(SRC_DIRS)
	$(POETRY) run black $(SRC_DIRS)
	$(POETRY) run isort $(SRC_DIRS)

mypy: install_dev
	$(POETRY) run mypy --show-error-codes $(MYPY_DIRS) $(ARGS)

lint: install_dev
	$(POETRY) check --lock
	$(POETRY) run flake8 $(SRC_DIRS)
	$(POETRY) run black $(SRC_DIRS) --check
	$(POETRY) run isort $(SRC_DIRS) --check-only --diff
	$(POETRY) run autoflake $(SRC_DIRS) --check
	# for now we only check data abstractions and services packages
	$(POETRY) run mypy --show-error-codes $(MYPY_DIRS)
	$(POETRY) run python scripts/check_imports.py

statgpt_cli: install_dev
	$(POETRY) run python -m statgpt.cli $(ARGS)

statgpt_admin:
	$(POETRY) run python -m statgpt.admin.app $(ARGS)

statgpt_fix_statuses:
	$(POETRY) run python -m statgpt.admin.fix_statuses

statgpt_auto_update:
	$(POETRY) run python -m statgpt.admin.auto_update

statgpt_app:
	$(POETRY) run python -m statgpt.app.app $(ARGS)

install_pre_commit_hooks:
	$(POETRY) run pre-commit install

db_migrate:
	$(POETRY) run alembic -c alembic.ini upgrade head

db_downgrade:
	$(POETRY) run alembic -c alembic.ini downgrade -1

db_autogenerate:
	$(POETRY) run alembic -c alembic.ini revision --autogenerate -m "$(MESSAGE)"

test_db_migrate: export PGVECTOR_HOST=$(TEST_DATABASE_HOST)
test_db_migrate: export PGVECTOR_PORT=$(TEST_DATABASE_PORT)
test_db_migrate: export PGVECTOR_DATABASE=$(TEST_DATABASE)
test_db_migrate: export ELASTIC_CONNECTION_STRING=$(TEST_ELASTIC_CONNECTION_STRING)
test_db_migrate: install_dev
	$(POETRY) run alembic -c alembic.ini upgrade head

test_unit: export PGVECTOR_HOST=$(TEST_DATABASE_HOST)
test_unit: export PGVECTOR_PORT=$(TEST_DATABASE_PORT)
test_unit: export PGVECTOR_DATABASE=$(TEST_DATABASE)
test_unit: export ELASTIC_CONNECTION_STRING=$(TEST_ELASTIC_CONNECTION_STRING)
test_unit: install_dev
	$(POETRY) run pytest tests/unit --junitxml=reports/tests-unit.xml

test_integration: export EMBEDDING_DEFAULT_MODEL=text-embedding-3-large
test_integration: export PGVECTOR_HOST=$(TEST_DATABASE_HOST)
test_integration: export PGVECTOR_PORT=$(TEST_DATABASE_PORT)
test_integration: export PGVECTOR_DATABASE=$(TEST_DATABASE)
test_integration: export ELASTIC_CONNECTION_STRING=$(TEST_ELASTIC_CONNECTION_STRING)
test_integration: test_db_migrate
	$(POETRY) run pytest tests/integration --junitxml=reports/tests-int.xml

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
	@cd statgpt/app/utils/formatters && \
	xgettext -d dataset -o locales/dataset.pot \
		--language=Python \
		--keyword=_ \
		--from-code=UTF-8 \
		base.py dataset_base.py dataset_simple.py dataset_detailed.py datasets_list_formatter.py citation.py \
		dataset_query.py dataset_availablity_query.py

update_messages: check_gettext
	@echo "Updating .po files from template..."
	@cd statgpt/app/utils/formatters/locales && \
	msgmerge --update en/LC_MESSAGES/dataset.po dataset.pot && \
	msgmerge --update uk/LC_MESSAGES/dataset.po dataset.pot

compile_messages: check_gettext
	@echo "Compiling .po files to .mo files..."
	@cd statgpt/app/utils/formatters/locales && \
	msgfmt -o en/LC_MESSAGES/dataset.mo en/LC_MESSAGES/dataset.po && \
	msgfmt -o uk/LC_MESSAGES/dataset.mo uk/LC_MESSAGES/dataset.po

# Convenience command to compile messages after changes
locales: compile_messages

# Utility to generate UUIDs
generate_uuid:
	python -c "from uuid import uuid4; print(uuid4())"
