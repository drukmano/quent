# quent Examples

Runnable cookbook recipes demonstrating quent's fluent pipeline builder and its transparent sync/async bridging.

## Prerequisites

```bash
pip install quent
```

Or, from the project root (development install):

```bash
pip install -e .
```

## How to Run

```bash
python examples/<recipe>.py
```

Each script is self-contained and uses only quent + the standard library.

## Recipes

| File | Description | Key Features |
|------|-------------|--------------|
| [`etl_pipeline.py`](etl_pipeline.py) | Complete Extract-Transform-Load pipeline | `then`, `do`, `foreach`, `except_`, `finally_`, `clone` |
| [`api_gateway.py`](api_gateway.py) | API gateway with auth, routing, and error handling | `gather`, `if_`/`else_`, `do`, `except_`, `Chain.return_` |
| [`fan_out_fan_in.py`](fan_out_fan_in.py) | Fan-out/fan-in concurrent processing | `gather`, `concurrency`, `clone`, sync and async variants |
| [`retry_backoff.py`](retry_backoff.py) | Retry with exponential backoff and jitter | `except_`, nested chains, sync and async retry wrappers |
| [`testing_chains.py`](testing_chains.py) | Unit testing quent chains with `unittest` | `MagicMock(spec=...)`, `AsyncMock`, `clone`, `decorator`, `on_step` |

## Notes

- All recipes demonstrate quent's transparent sync/async bridging: the same pipeline definition works with both sync and async callables.
- No external dependencies are required beyond quent itself.
