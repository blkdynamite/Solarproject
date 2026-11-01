.PHONY: install lint typecheck test check fleet fleet-quick solar evaluate clean

install:            ## editable install with dev tools, pinned to tested versions
	pip install -c requirements-lock.txt -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src tests

test:
	pytest --cov=solar_rooftop --cov=der_mv --cov-report=term-missing

check: lint typecheck test   ## everything CI runs

fleet:              ## DR M&V study + 20-fleet validation -> results/fleet_mv/ (~15 min)
	fleet-mv run --config config/fleet_thermostat_dr.toml --out results/fleet_mv --validate 20

fleet-quick:        ## DR M&V study without the multi-fleet validation (~40 s)
	fleet-mv run --config config/fleet_thermostat_dr.toml --out results/fleet_mv

solar:              ## rooftop solar analysis -> results/
	solar-rooftop run --config config/greensboro_300kw.toml --out results

evaluate:           ## Monte Carlo test of the fault detector
	solar-rooftop evaluate-detector --seeds 200

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage build src/*.egg-info
