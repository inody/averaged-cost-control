SHELL := /bin/bash

FIGURE_OUTPUT ?= outputs/frozen-figures
FULL_OUTPUT ?= outputs/full-reproduction
BENCHMARK_OUTPUT ?= outputs/runtime-benchmark
SENSITIVITY_OUTPUT ?= $(FULL_OUTPUT)/sensitivity

.PHONY: setup figures paper-figures reproduce recompute sensitivity sensitivity-figure full-recompute runtime-figure benchmark test verify verify-results clean

setup:
	uv sync --frozen --extra dev

figures:
	@test ! -e "$(FIGURE_OUTPUT)" || { echo "Output already exists: $(FIGURE_OUTPUT)"; exit 1; }
	uv run python -m experiments.frozen_figures --output "$(FIGURE_OUTPUT)"

paper-figures: figures

recompute:
	@test ! -e "$(FULL_OUTPUT)/joint-study" || { echo "Output already exists: $(FULL_OUTPUT)/joint-study"; exit 1; }
	@mkdir -p "$(FULL_OUTPUT)"
	uv run python -m experiments.joint_uncertainty.run --output "$(FULL_OUTPUT)/joint-study"
	uv run python -m experiments.verify_results \
		--summary "$(FULL_OUTPUT)/joint-study/summary.json" \
		--figures "$(FULL_OUTPUT)/joint-study/figures"

sensitivity:
	@test -f "$(FULL_OUTPUT)/joint-study/summary.json" || { echo "Run make recompute first."; exit 1; }
	uv run python -m experiments.uncertainty_sensitivity.run \
		--output "$(SENSITIVITY_OUTPUT)" \
		--source-summary "$(FULL_OUTPUT)/joint-study/summary.json"
	uv run python -m experiments.verify_release \
		--sensitivity "$(SENSITIVITY_OUTPUT)/summary.json" \
		--reference-sensitivity data/reference_sensitivity.json \
		--points "$(SENSITIVITY_OUTPUT)/points"

reproduce:
	@test ! -e "$(FULL_OUTPUT)" || { echo "Output already exists: $(FULL_OUTPUT)"; exit 1; }
	$(MAKE) recompute FULL_OUTPUT="$(FULL_OUTPUT)"
	$(MAKE) sensitivity FULL_OUTPUT="$(FULL_OUTPUT)" SENSITIVITY_OUTPUT="$(SENSITIVITY_OUTPUT)"
	@mkdir -p "$(FULL_OUTPUT)/figures"
	@cp "$(FULL_OUTPUT)"/joint-study/figures/*.pdf "$(FULL_OUTPUT)/figures/"
	uv run python -m experiments.initial_state_scaling.render_recorded \
		--output "$(FULL_OUTPUT)/figures/initial-state-runtime-scaling.pdf"
	uv run python -m experiments.uncertainty_sensitivity.render \
		--summary "$(SENSITIVITY_OUTPUT)/summary.json" \
		--output "$(FULL_OUTPUT)/figures/uncertainty-variance-cost-sensitivity.pdf"
	uv run python -m experiments.paper_correspondence \
		--results "$(FULL_OUTPUT)/joint-study/summary.json" \
		--output "$(FULL_OUTPUT)/table1-rows.tex"
	uv run python -m experiments.verify_release \
		--figures "$(FULL_OUTPUT)/figures" \
		--results "$(FULL_OUTPUT)/joint-study/summary.json" \
		--sensitivity "$(SENSITIVITY_OUTPUT)/summary.json" \
		--reference-sensitivity data/reference_sensitivity.json \
		--points "$(SENSITIVITY_OUTPUT)/points"

full-recompute: reproduce

sensitivity-figure:
	@test ! -e "$(FIGURE_OUTPUT)/uncertainty-variance-cost-sensitivity.pdf" || { echo "Output already exists."; exit 1; }
	uv run python -m experiments.uncertainty_sensitivity.render \
		--summary data/reference_sensitivity.json \
		--output "$(FIGURE_OUTPUT)/uncertainty-variance-cost-sensitivity.pdf"

runtime-figure:
	@test ! -e "$(FIGURE_OUTPUT)/initial-state-runtime-scaling.pdf" || { echo "Output already exists."; exit 1; }
	uv run python -m experiments.initial_state_scaling.render_recorded \
		--output "$(FIGURE_OUTPUT)/initial-state-runtime-scaling.pdf"

benchmark:
	@test -f "$(FULL_OUTPUT)/joint-study/summary.json" || { echo "Run make recompute first."; exit 1; }
	@test ! -e "$(BENCHMARK_OUTPUT)" || { echo "Output already exists: $(BENCHMARK_OUTPUT)"; exit 1; }
	uv run python -m experiments.initial_state_scaling.benchmark \
		--output "$(BENCHMARK_OUTPUT)" \
		--joint-summary "$(FULL_OUTPUT)/joint-study/summary.json"

test:
	PYTHONDONTWRITEBYTECODE=1 uv run pytest

verify: test
	uv run python -m experiments.frozen_figures --verify-manifest
	@set -eu; temporary=$$(mktemp -d); \
	trap 'rm -rf "$$temporary"' EXIT; \
	uv run python -m experiments.frozen_figures --output "$$temporary/figures"; \
	uv run python -m experiments.verify_release --figures "$$temporary/figures"

verify-results:
	@test -f "$(FULL_OUTPUT)/joint-study/summary.json" || { echo "Run make recompute first."; exit 1; }
	uv run python -m experiments.verify_results \
		--summary "$(FULL_OUTPUT)/joint-study/summary.json" \
		--figures "$(FULL_OUTPUT)/joint-study/figures"

clean:
	rm -rf outputs
