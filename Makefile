SHELL := /bin/bash

PAPER_OUTPUT ?= outputs/paper-figures
BENCHMARK_OUTPUT ?= outputs/runtime-benchmark
SENSITIVITY_OUTPUT ?= outputs/uncertainty-sensitivity

.PHONY: setup figures paper-figures recompute sensitivity sensitivity-figure full-recompute runtime-figure benchmark test verify verify-results clean

setup:
	uv sync --frozen --extra dev

figures: paper-figures

paper-figures:
	@test ! -e "$(PAPER_OUTPUT)" || { echo "Output already exists: $(PAPER_OUTPUT)"; exit 1; }
	uv run python -m experiments.joint_uncertainty.run \
		--output "$(PAPER_OUTPUT)/joint-study"
	uv run python -m experiments.verify_results \
		--summary "$(PAPER_OUTPUT)/joint-study/summary.json" \
		--figures "$(PAPER_OUTPUT)/joint-study/figures"
	uv run python -m experiments.initial_state_scaling.render_recorded \
		--output "$(PAPER_OUTPUT)/initial-state-runtime-scaling.pdf"
	uv run python -m experiments.uncertainty_sensitivity.render \
		--summary data/reference_sensitivity.json \
		--output "$(PAPER_OUTPUT)/uncertainty-variance-cost-sensitivity.pdf"
	@mkdir -p "$(PAPER_OUTPUT)/figures"
	@cp "$(PAPER_OUTPUT)"/joint-study/figures/*.pdf "$(PAPER_OUTPUT)/figures/"
	@cp "$(PAPER_OUTPUT)/initial-state-runtime-scaling.pdf" "$(PAPER_OUTPUT)/figures/"
	@cp "$(PAPER_OUTPUT)/uncertainty-variance-cost-sensitivity.pdf" "$(PAPER_OUTPUT)/figures/"
	@echo "Publication figures: $(PAPER_OUTPUT)/figures"

recompute:
	@test ! -e "$(PAPER_OUTPUT)/joint-study" || { echo "Output already exists: $(PAPER_OUTPUT)/joint-study"; exit 1; }
	uv run python -m experiments.joint_uncertainty.run \
		--output "$(PAPER_OUTPUT)/joint-study"

sensitivity:
	@test -f "$(PAPER_OUTPUT)/joint-study/summary.json" || { echo "Run make recompute first."; exit 1; }
	uv run python -m experiments.uncertainty_sensitivity.run \
		--output "$(SENSITIVITY_OUTPUT)" \
		--source-summary "$(PAPER_OUTPUT)/joint-study/summary.json"
	uv run python -m experiments.uncertainty_sensitivity.render \
		--summary "$(SENSITIVITY_OUTPUT)/summary.json" \
		--output "$(SENSITIVITY_OUTPUT)/uncertainty-variance-cost-sensitivity.pdf"

sensitivity-figure:
	uv run python -m experiments.uncertainty_sensitivity.render \
		--summary data/reference_sensitivity.json \
		--output "$(PAPER_OUTPUT)/uncertainty-variance-cost-sensitivity.pdf"

full-recompute:
	$(MAKE) recompute
	$(MAKE) sensitivity

runtime-figure:
	uv run python -m experiments.initial_state_scaling.render_recorded \
		--output "$(PAPER_OUTPUT)/initial-state-runtime-scaling.pdf"

benchmark:
	@test -f "$(PAPER_OUTPUT)/joint-study/summary.json" || { echo "Run make recompute first."; exit 1; }
	@test ! -e "$(BENCHMARK_OUTPUT)" || { echo "Output already exists: $(BENCHMARK_OUTPUT)"; exit 1; }
	uv run python -m experiments.initial_state_scaling.benchmark \
		--output "$(BENCHMARK_OUTPUT)" \
		--joint-summary "$(PAPER_OUTPUT)/joint-study/summary.json"

test:
	PYTHONDONTWRITEBYTECODE=1 uv run pytest

verify: test

verify-results:
	@test -f "$(PAPER_OUTPUT)/joint-study/summary.json" || { echo "Run make recompute first."; exit 1; }
	uv run python -m experiments.verify_results \
		--summary "$(PAPER_OUTPUT)/joint-study/summary.json" \
		--figures "$(PAPER_OUTPUT)/joint-study/figures"

clean:
	rm -rf outputs
