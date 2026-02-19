PYTHON ?= /data/yiming/conda_envs/mle_master/bin/python

.PHONY: env-check run test

env-check:
	@test -x "$(PYTHON)" || (echo "Python interpreter not found: $(PYTHON)" && exit 1)

run: env-check
	$(PYTHON) main.py

test: env-check
	$(PYTHON) -m pytest -q
