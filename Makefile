.PHONY: install lint typecheck test check run evaluate clean

install:            ## editable install with dev tools, pinned to tested versions
	pip install -c requirements-lock.txt -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src tests

test:
	pytest --cov=solar_rooftop --cov-report=term-missing

check: lint typecheck test   ## everything CI runs

run:                ## full analysis -> results/
	solar-rooftop run --config config/greensboro_300kw.toml --out results

evaluate:           ## Monte Carlo test of the fault detector
	solar-rooftop evaluate-detector --seeds 200

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage build src/*.egg-info
