format:
	uv run isort . && uv run ruff format

check:
	uv run ruff check --fix
