# Contributing to quent

## Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) for dependency management (recommended), or pip

## Getting Started

```bash
git clone https://github.com/drukmano/quent.git
cd quent
uv sync --group dev
bash scripts/setup.sh  # Install pre-commit hook
```

**Without uv:** If you do not have `uv` installed, you can use pip instead:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
pip install ruff mypy pip-audit hypothesis
bash scripts/setup.sh
```

## Running Tests

Full QA pipeline (format, lint, type check, tests):

```bash
bash run_tests.sh
```

Run a single test file:

```bash
python -m unittest tests.null_tests
```

Run a single test method:

```bash
python -m unittest tests.null_tests.NullVsNoneTest.test_q_no_arg_has_no_root_value
```

## Code Style

- **2-space indentation**, enforced by ruff
- **120-character line length**
- **Single quotes** for strings

Format and lint before committing:

```bash
ruff format quent/
ruff check --fix quent/
```

## Type Checking

```bash
mypy quent/
```

## Reporting Issues

- **Bug reports:** Use the [bug report template](https://github.com/drukmano/quent/issues/new?template=bug_report.yml).
- **Feature requests:** Use the [feature request template](https://github.com/drukmano/quent/issues/new?template=feature_request.yml).

## Project Scope

quent is a **transparent sync/async bridge** -- it provides the minimum set of pipeline primitives that let developers write code once for both sync and async callables. When evaluating contributions, keep in mind:

- Every feature must justify itself by solving a real sync/async bridging problem or eliminating genuine code duplication.
- **The litmus test:** would removing the feature force users to write separate sync and async code paths? If the same user code works identically regardless of sync/async without the feature, it doesn't belong.
- quent is NOT a collections library, functional programming toolkit, or opinionated framework.
- Features that merely wrap stdlib functionality don't belong.

For more on the design philosophy, see [Why quent](https://quent.readthedocs.io/en/latest/why-quent/).

## Pull Request Guidelines

- One feature or fix per PR.
- Include tests for new behavior.
- Ensure the full QA pipeline passes (`bash run_tests.sh`).
- Use concise, imperative-mood commit messages (e.g., "Add gather exception handling").
- If your change is breaking, describe the impact in the PR description.
