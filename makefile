test:
	python -m pytest --cov=qcifc --cov-report="html" tests 2>&1 | tee errors.err
debug:
	python -m pytest --pdb tests
testv:
	python -m pytest -v --cov=qcifc --cov-report="html" tests| tee errors.err
testx:
	python -m pytest -x --cov=qcifc --cov-report="html" tests| tee errors.err
