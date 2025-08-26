.PHONY: qa fmt type test

qa: fmt type test
fmt:
	ruff check . && black --check .
type:
	mypy src/
test:
	pytest -q