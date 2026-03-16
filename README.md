# StatGPT Backend

This repository contains code for StatGPT backend, which implements APIs and main logic of the StatGPT application.

More information about StatGPT and its architecture can be found in
the [documentation repository](https://github.com/epam/statgpt).

## Technological stack

Application is written in Python 3.11 and uses the following main technologies:

| Technology                                                   | Purpose                                                  |
|--------------------------------------------------------------|----------------------------------------------------------|
| [AI DIAL SDK](https://github.com/epam/ai-dial-sdk)           | SDK for building applications on top of AI DIAL platform |
| [FastAPI](https://fastapi.tiangolo.com/)                     | Web framework for API development                        |
| [SQLAlchemy](https://www.sqlalchemy.org/)                    | ORM for database operations                              |
| [LangChain](https://python.langchain.com/docs/introduction/) | LLM application framework                                |
| [Pydantic](https://pydantic.dev/)                            | Data validation and settings                             |
| [sdmx1](https://github.com/khaeru/sdmx)                      | SDMX data handling and provider connections              |

## Project structure

* `statgpt/admin` — backend of the administrator part which allows the user to add and update data.
* `statgpt/common` — common code used in the `statgpt.admin` and `statgpt.app` applications.
* `statgpt/app` — main application that generates response using LLMs and based on data prepared by `statgpt.admin`.
* `statgpt/cli` — command-line interface (CLI) for managing various aspects of StatGPT.
* `tests` - unit and integration tests.
* `docker` - Dockerfiles for building docker images.

## Environment variables

The applications are configured using environment variables. The environment variables are described in the following
files:

* [Common environment variables](statgpt/common/README.md#environment-variables) - used in both applications
* [Admin Backend environment variables](statgpt/admin/README.md#environment-variables)
* [Main App environment variables](statgpt/app/README.md#environment-variables)
* [CLI environment variables](statgpt/cli/README.md#environment-variables)

## Local Setup

### Pre-requisites

#### 1. Install [Make](https://www.gnu.org/software/make/)

* MacOS - should be already installed
* [Windows](https://gnuwin32.sourceforge.net/packages/make.htm)
* [Windows, using Chocolatey](https://community.chocolatey.org/packages/make)
* Make sure that `make` is in the PATH (run `which make`).

#### 2. Install Python 3.11

Direct installation:

* [MacOS, using Homebrew](https://formulae.brew.sh/formula/python@3.11) - `brew install python@3.11`
* [Windows or MacOS, using official repository](https://www.python.org/downloads/)
* [Windows, using Chocolatey](https://community.chocolatey.org/packages/python311)
* Make sure that `python3` or `python3.11` is in the PATH and works properly (run `python3.11 --version`).

Alternative: use [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation):

* `pyenv` allows to manage different python versions on the same machine
* execute following from the repository root folder:
  ```bash
  pyenv install 3.11
  pyenv local 3.11  # use Python 3.11 for the current project
  ```

#### 3. Install [Poetry](https://python-poetry.org/docs/#installation)

Recommended way - system-wide, independent of any particular python venv:

* MacOS - recommended way to install poetry is to [use pipx](https://python-poetry.org/docs/#installing-with-pipx)
* Windows - recommended way to install poetry is to
  use [official installer](https://python-poetry.org/docs/#installing-with-the-official-installer)
* Make sure that `poetry` is in the PATH and works properly (run `poetry --version`).

#### 4. Install Docker Engine and Docker Compose suitable for your OS

Since Docker Desktop requires a paid license for commercial use, you can use one of the following alternatives:

* [Docker Engine and Docker Compose on Linux](https://docs.docker.com/engine/install/)
* [Rancher Desktop](https://rancherdesktop.io/) on Windows or MacOS

#### 5. Install GNU gettext (for localization)

Required for localization commands (`make extract_messages`, `make update_messages`, `make compile_messages`):

* MacOS - `brew install gettext`
* Linux/WSL - `sudo apt install gettext`
* Windows (native) - Install via [Chocolatey](https://community.chocolatey.org/packages/gettext): `choco install gettext`

Verify installation: `which xgettext msgmerge msgfmt`

---

### Setup

#### 1. Clone the repository

#### 2. Create venv (python virtual environment)

Create python virtual environment, using poetry:

```bash
make init_venv
```

If you see the following error: `Skipping virtualenv creation, as specified in config file.`, it means venv was not
created because poetry is configured not to create a new virtual environment. You can fix this:

* Either by updating poetry config:
    * `poetry config --local virtualenvs.create true` (local config)
    * or `poetry config virtualenvs.create true` (global config)
* or by creating venv manually: `python -m venv .venv`

#### 3. Activate venv

For Mac / Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv/Scripts/Activate
```

#### 4. Install required python packages

The following will install basic and dev dependencies:

```bash
make install_dev
```

#### 5. Create `.env` file in the root of the project

You can copy the template file and fill values for secrets manually:

```bash
cp .env.template .env
```

The [Environment variables section](#environment-variables) provides links to pages with
detailed information about environment variables.

#### 6. Create `dial/core/config/config.json` file by running python script

_Not implemented yet, TODO: create a script that generates config based on .env variables_

## Run StatGPT locally

1. Run the DIAL using docker compose:

    ```bash
    docker compose up -d
    ```

2. Apply `alembic` migrations:

   ```bash
   make db_migrate
   ```

3. Run Admin backend (if you want to initialize or update data):

   ```bash
   make statgpt_admin
   ```

4. Run StatGPT application:

   ```bash
   make statgpt_app
   ```

5. Initialize sample content (optional):

   ```bash
   # Run CLI and initialize sample client
   make statgpt_cli
   ```

   Then in the CLI:

   ```
   statgpt> content init --client-id sample -y
   statgpt> channel reindex -c statgpt-sample --mode all
   ```

   Wait till reindexing is finished (check status using `channel status` command in CLI). After that run deduplication:

   ```
   statgpt> channel deduplicate -c statgpt-sample
   ```


   See [CLI documentation](statgpt/cli/README.md) for more commands.

## Admin MCP (Beta)

The Admin application includes an optional MCP (Model Context Protocol) server for dataset onboarding assistance.
It provides tools and prompts for coding agents such as Cursor and Claude Code.

> **Note:** This feature is optional and disabled by default. It requires installing additional dependencies
> and enabling via environment variable.

See [MCP setup instructions](statgpt/admin/mcp/README.md) for details.

## Utils for Development

### 1. Format the code

 ```bash
 make format
 ```

### 2. Run linters

 ```bash
 make lint
 ```

### 3. Pre-Commit Hooks

To automatically apply black and isort on each commit, enable PreCommit Hooks:

```bash
make install_pre_commit_hooks
```

This command will set up the git hook scripts.

### 4. Create a new `alembic` migration:

> **(!)**
> It is critical to note that **autogenerate is not intended to be perfect**.
> It is *always* necessary to manually review and correct the **candidate migrations** that autogenerate produces.

**(!)** After creating a new migration, it is necessary to update the `ALEMBIC_TARGET_VERSION` in the
`statgpt/common/config/version.py` file to the new version.

 ```bash
 make db_autogenerate MESSAGE="Your message"
 ```

or:

 ```bash
 alembic -c alembic.ini revision --autogenerate -m "Your message"
 ```

### 5. Undo last `alembic` migration

 ```bash
 make db_downgrade
 ```

### 6. Localization (i18n)

The project uses GNU gettext for internationalizing dataset formatters. Use these commands when working with translations:

**Workflow:**

1. **Extract translatable strings** - Run after adding/modifying strings marked with `_()` in formatter code:
   ```bash
   make extract_messages
   ```
   This creates/updates the `locales/dataset.pot` template file.

2. Review changes to `locales/dataset.pot` file - check git diff.
   There should be no unexpected changes (removals, additions) -
   they sometimes happen on Windows platforms.

3. **Update translation files** - Run to sync `.po` files with the new template:
   ```bash
   make update_messages
   ```
   This updates `en/LC_MESSAGES/dataset.po` and `uk/LC_MESSAGES/dataset.po` with new strings.

4. Fill missing translations in `.po` files. Either manually or using coding agent.

5. **Compile translations** - Run after translating strings in `.po` files to generate binary `.mo` files:
   ```bash
   make compile_messages
   ```
   Or use the shorthand: `make locales`

**Note:** All commands require GNU gettext to be installed (see [Prerequisites](#pre-requisites)).

## Run Tests

- Run all tests (unit and integration):

    ```bash
    make test
    ```

- Run only unit tests:

    ```bash
    make test_unit
    ```

- Run only integration tests:

    ```bash
    make test_integration
    ```

⚠️ **WARNING:** Integration tests require a database and Elasticsearch instance.
Consider using separate test instances instead of the ones from `docker-compose.yml`
because tests truncate tables during execution, which may result in **DATA LOSS**.
Configure `TEST_DATABASE_*` environment variables accordingly.
See [Common environment variables](statgpt/common/README.md#environment-variables) for details.
