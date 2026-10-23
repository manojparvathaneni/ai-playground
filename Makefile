# Shortcuts for common tasks. Run `make` on its own to list them.
# Each one is a plain command you can also run by hand; see the lines below.

.DEFAULT_GOAL := help
.PHONY: help install check lint format typecheck test lock up down backup restore requirements

help:  ## List the available commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-13s %s\n", $$1, $$2}'

install:  ## Create .venv with all dependencies, including the dev tools
	uv sync

check: lint typecheck test lock  ## Run every check that CI runs. Do this before committing.

lint:  ## Look for likely bugs and style problems, and check formatting
	uv run ruff check
	uv run ruff format --check

format:  ## Fix formatting and the problems ruff can fix by itself
	uv run ruff format
	uv run ruff check --fix

typecheck:  ## Check the type hints in src/ and tests/
	uv run mypy

test:  ## Run the tests (offline: no model server or Docker needed)
	uv run pytest

lock:  ## Check that uv.lock and requirements.txt match pyproject.toml
	uv lock --check
	@$(REQUIREMENTS) | diff -q - requirements.txt > /dev/null \
		|| { echo "requirements.txt is out of date: run 'make requirements'"; exit 1; }

requirements:  ## Regenerate requirements.txt from uv.lock
	@$(REQUIREMENTS) > requirements.txt

# Prints the contents requirements.txt should have.
REQUIREMENTS = { \
	echo '\# Generated from uv.lock for tools that need a requirements file.'; \
	echo '\# Do not edit by hand: change pyproject.toml, then run `make requirements`.'; \
	echo '\# With uv, just run `uv sync` instead.'; \
	uv export --no-hashes --no-emit-project --no-header --no-dev --quiet; \
}

up:  ## Start Phoenix and Langfuse in Docker
	docker compose up -d

down:  ## Stop Phoenix and Langfuse (traces are kept)
	docker compose down

backup:  ## Save all Phoenix and Langfuse data to backups/<date-time>/
	docker/backup.sh

restore:  ## Replace the data with a backup: make restore BACKUP=backups/<date-time>
	docker/restore.sh $(BACKUP)
