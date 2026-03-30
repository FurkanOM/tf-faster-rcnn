.PHONY: lint test check

lint:
	python tools/lint.py

test:
	python -m unittest discover -s tests -p 'test_*.py'

check:
	python tools/run_checks.py
