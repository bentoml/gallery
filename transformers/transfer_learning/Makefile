format:
	black --config ../../pyproject.toml .
	isort .

test:
	pytest -vvv

run:
	bentoml serve service.py:svc --reload

benchmark:
	locust --headless -u 100 -r 1000 --run-time 2m --host http://127.0.0.1:3000

new_nb:
	python -m jupyter_ascending.scripts.make_pair --base $(filename)
