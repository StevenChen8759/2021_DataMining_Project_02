
init:
	pipenv --three
	pipenv install
	pipenv install --dev

run:
	pipenv run python ./src/main.py

commit:
	pipenv run cz commit

lint:
	pipenv run flake8
	pipenv run pylint
