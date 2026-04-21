# Contributing to Geo-BTC

Thank you for your interest in contributing. Geo-BTC is a scientific repository
released as a companion to a peer-reviewed article, so contributions are held to
standards of **reproducibility**, **traceability**, and **scientific rigour**.

## Ways to contribute

- Report bugs or reproducibility issues by opening an issue.
- Improve documentation (typos, clarifications, additional context).
- Add tests that harden the existing pipeline.
- Propose new features that are clearly related to the scope of the paper.

## Development workflow

1. Fork the repository and create a topic branch:
   ```bash
   git checkout -b fix/descriptive-name
   ```
2. Install the project in editable mode with development dependencies:
   ```bash
   make install
   ```
3. Ensure style and tests pass before pushing:
   ```bash
   make lint
   make test
   ```
4. Commit with a descriptive message. We prefer
   [Conventional Commits](https://www.conventionalcommits.org/) prefixes
   (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
5. Open a pull request referencing the related issue.

## Code style

- Python 3.10+ with type hints where reasonable.
- Formatter and linter: [`ruff`](https://docs.astral.sh/ruff/) (`make format`).
- Docstrings follow the NumPy / SciPy convention.
- Keep functions small, testable, and side-effect free whenever possible.

## Reproducibility expectations

Every contribution that changes the scientific pipeline must:

- Preserve deterministic behaviour (random seeds from `src/utils/seed.py`).
- Keep `scripts/run_pipeline.py` functional end-to-end.
- Include an updated unit test if it modifies a function in `src/`.
- Declare any new dependency in both `requirements.txt` and `pyproject.toml`.

## Reporting scientific errors

If you believe you have identified a methodological error that affects results
reported in the paper, please open an issue tagged `scientific-error` and
include:

- The exact commit hash you reproduced.
- A minimal script or notebook demonstrating the discrepancy.
- The observed vs expected behaviour.

Such reports will be handled with the highest priority.

## Code of conduct

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
