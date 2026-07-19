.PHONY: install check smoke generate score test lint

install:
	python -m pip install -r requirements-gpu.txt

check:
	python -m generation.check_gpu

smoke:
	python -m generation.run_model --config configs/smoke.yaml

generate:
	python -m generation.generate_dataset --config configs/math500_100.yaml

score:
	python -m evaluation.score_results results/math500_100.jsonl

test:
	python -m pytest

lint:
	python -m ruff check .

