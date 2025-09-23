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

* `src/admin_portal` — backend of the administrator portal which allows the user to add and update data.
* `src/common` — common code used in the `admin_portal` and `statgpt` applications.
* `src/statgpt` — main application that generates response using LLMs and based on data prepared by `admin_portal`.
* `tests` - unit and integration tests.
* `docker` - Dockerfiles for building docker images.

## Environment variables

The applications are configured using environment variables. The environment variables are described in the following
files:

* [Common environment variables](src/common/README.md#environment-variables) - used in both applications
* [Admin Backend environment variables](src/admin_portal/README.md#environment-variables)
* [Chat Backend environment variables](src/statgpt/README.md#environment-variables)

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

Alternative - venv-specific (using `pip`):

* make sure the correct python venv is activated
* `make install_poetry`

#### Install Docker Engine and Docker Compose suitable for your OS

Since Docker Desktop requires a paid license for commercial use, you can use one of the following alternatives:

* [Docker Engine and Docker Compose on Linux](https://docs.docker.com/engine/install/)
* [Rancher Desktop](https://rancherdesktop.io/) on Windows or MacOS

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

#### 6. Create `dial_conf/core/config.json` file by running python script

```bash
make generate_dial_config
```

## Run StatGPT locally

1. Run the DIAL using docker compose:

    ```bash
    docker compose up -d
    ```

2. Apply `alembic` migrations:
    * locally:

        ```bash
        make db_migrate
        ```

    * or using Docker:
        1) Set `ADMIN_MODE=ALEMBIC_UPGRADE` in the `.env` file
        2) Run `admin_portal` from `docker-compose.yml`

3. Run Admin backend (if you want to initialize or update data):

    ```bash
    make run_admin
    ```

4. There are two ways to initialize data:
    1) Using import feature (A zip file with the exported channel is required):
        1. Uncomment and run the `admin-ui` service in the `docker-compose.yml` file.
        2. Open the [admin portal](http://localhost:4100/channels) in your browser and navigate to the `Channels` tab.
        3. Click on the `Import` button and select the `<exported channel>.zip` file.

    2) With predefined content in `scripts/config/*.yaml` files:

        ```bash
        make init_content
        ```

       Note:
        * Use `MAX_N_EMBEDDINGS` env var to limit the maximum number of computed embeddings per collection
          (for test/debug purpose)
        * Use the following command to get the current preprocessing statuses of datasets and channel datasets:

          ```bash
          make status_monitor
          ```

5. Now you can stop the applications/containers from the previous two steps (4-5), they are no longer needed.
   Run Chat backend:

    ```bash
    make run_chat
    ```

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
`src/common/config/version.py` file to the new version.

 ```bash
 make db_autogenerate MESSAGE="Your message"
 ```

or:

 ```bash
 alembic -c src/alembic.ini revision --autogenerate -m "Your message"
 ```

### 5. Undo last `alembic` migration

 ```bash
 make db_downgrade
 ```

### 6 Localizations

To update localization files, run:

```bash
make extract_messages  # Extract messages to be translated
make update_messages   # Update existing translation files with new messages
```

Check the *.po files for new messages and provide translations.

Then compile translations:

```bash
make compile_messages
```

## Run Tests

1. Integration tests require running a test database and elasticsearch.
   They are part of the `docker-compose.yml` file.
   The Docker containers with this database/elasticsearch don't have volumes to store data,
   so they are always fresh after `docker compose down`.
2. To run integration tests, uncomment the `vectordb-test` and `elasticsearch-test` containers in the
   `docker-compose.yml` file.
   You might also need to comment out the `elasticsearch` container if your machine doesn't have enough resources.
3. To run end-to-end tests, first run StatGPT locally. This step is not required for other tests.
4. Run tests:
    * all tests except for end-to-end (unit and integration):

        ```bash
        make test
        ```

    * only unit tests:

        ```bash
        make test_unit
        ```

    * only integration tests:

        ```bash
        make test_integration
        ```

    * just end-to-end tests:

        ```bash
        make test_e2e
        ```
