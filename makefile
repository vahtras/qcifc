test:
	python -m pytest --cov=qcifc --cov-report="term" --cov-report="html" tests 2>&1 | tee errors.err
debug:
	python -m pytest -x --pdb tests
testv:
	python -m pytest -v --cov=qcifc --cov-report="html" tests| tee errors.err
testx:
	python -m pytest -x --cov=qcifc --cov-report="html" tests| tee errors.err
browse:
	cd htmlcov && python3 -c 'import webbrowser; webbrowser.open_new_tab("http://localhost:8000")' && python3 -m http.server 

