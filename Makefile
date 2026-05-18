setup:
	if not exist .env copy .env.example .env
	py -3.12 -m venv .venv
	.venv\Scripts\python -m pip install --upgrade pip
	.venv\Scripts\python -m pip install -r backend\requirements.txt
	.venv\Scripts\python -m pip install -r frontend\requirements.txt

up:
	docker compose up --build

down:
	docker compose down

ingest-samples:
	.venv\Scripts\python scripts\ingest_sample_docs.py

eval:
	.venv\Scripts\python scripts\run_eval.py

eval-fast:
	.venv\Scripts\python scripts\run_eval.py --fast --max-questions 8 --timeout 900

test:
	.venv\Scripts\python -m pytest backend\tests -q
