test:
	python -m pytest --cov=qcifc --cov-report="html" tests
testv:
	python -m pytest -v --cov=qcifc --cov-report="html" tests
testx:
	python -m pytest -x --cov=qcifc --cov-report="html" tests
