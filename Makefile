.PHONY: install migrate test demo eval lint format ui api

install:
	uv sync

migrate:
	uv run alembic upgrade head

test:
	uv run pytest

demo:
	uv run python -m app.scripts.demo

eval:
	uv run python -m eval.run

lint:
	uv run ruff check app tests

format:
	uv run ruff format app tests

ui:
	uv run streamlit run ui/streamlit_app.py

api:
	uv run uvicorn app.main:app --reload
