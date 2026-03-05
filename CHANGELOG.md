# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [3.1.0] - 2026-03-05

### Breaking
- **Pure Python rewrite**: Migrated from Cython-compiled C extensions to pure Python
- **Removed classes**: `Cascade`, `CascadeAttr`, `ChainAttr` — use `Chain` directly
- **Removed pipe operator**: The `|` syntax and `run` helper class are no longer available
- **Removed resilience features**: `retry()`, `timeout()`, `safe_run()` methods removed
- **Removed context propagation**: `with_context()`, `get_context()` methods removed
- **Removed methods**: `root()`, `root_do()`, `clone()`, `config()`, `autorun()`, `while_true()`, `sleep()`, `raise_()`, `attr()`, `attr_fn()`, `suppress()`, `on_success()`, `reduce()`, `set_async()`, `compose()`, `pipe()`, `Chain.null()`
- **Removed conditionals**: `if_()`, `else_()`, `if_not()`, `condition()`, `not_()`, `eq()`, `neq()`, `is_()`, `is_not()`, `in_()`, `not_in()`, `isinstance_()`, `or_()`, `if_raise()`, `else_raise()`, `if_not_raise()`
- **Simplified except_**: Only one handler per chain, no `reraise` parameter
- **Python 3.10 minimum**: Lowered from 3.14 to 3.10 (eager_start=True still used on 3.14+)

### Changed
- Core rewritten as five pure Python modules: `_chain.py`, `_core.py`, `_ops.py`, `_traceback.py`, `__init__.py`
- Link evaluation simplified: direct callable/value dispatch via `_evaluate_value()`
- Async detection uses `inspect.isawaitable()` instead of C-level type checks
- `_FrozenChain` is now a simple wrapper delegating to the underlying Chain

## [3.0.0] - 2026-03-01

### Breaking
- **Python 3.14 minimum**: Raised minimum Python version from 3.10 to 3.14
- **ExceptionGroup for retries**: Retry exhaustion now raises `ExceptionGroup` containing all individual failures instead of the last exception

### Added
- Eager task creation via `asyncio.create_task(eager_start=True)` for 2-5x faster chains with sync-completing coroutines
- `add_note()` chain visualization on exceptions (Python 3.11+)
- `co_qualname` support in traceback code objects
- `__repr__` and `__bool__` stubs in type annotations
- PEP 695 `type` statement syntax in type stubs and `__init__.py`

### Changed
- `link.result` assignment gated behind `debug` mode for hot path optimization
- `ContextVar` token context manager (Python 3.14) replaces manual token management in `_async_with_context`
- Removed defensive `try/except (ValueError, RuntimeError)` around context reset in `run()`
- `sys.exception()` replaces `sys.exc_info()` for modern exception access
- Bare `except:` replaced with `except BaseException:` in `foreach`
- `try/except AttributeError` replaced with `hasattr()` in `with_`
- Deprecated `asyncio.iscoroutine()` replaced with `inspect.iscoroutine()` in tests
- Type stubs modernized: `Optional[X]` → `X | None`, `Link` alias renamed to `ChainLink`
- Cython build dependency updated to `>=3.2.4`
- CI/CD updated for Python 3.14 only

### Removed
- Stale TODO comments in traceback handling
- pytest dependency in cibuildwheel (replaced with unittest)

## [2.2.0] - 2025-08-06

### Added
- `Chain.is_instance_()` method
- `.sleep()` method

### Changed
- Improved `__repr__` and exception formatting
- Minor optimizations
- Improved stub file
- Silenced compilation warnings

### Fixed
- Minor bug fixes

## [2.1.2] - 2023-12-05

### Fixed
- Minor fixes and cleanup

## [2.1.1] - 2023-11-30

### Changed
- Removed the `return_` argument for `.except_()`

## [2.1.0] - 2023-11-30

### Changed
- Exception handlers are now positional arguments

## [2.0.4] - 2023-11-30

### Added
- `.return_()` method
- `.break_()` method

## [2.0.3] - 2023-11-26

### Added
- `.while_true()` method

## [2.0.2] - 2023-11-25

### Added
- `return_` parameter for `.except_()`
- Allow a `Chain` without a root value

### Fixed
- Bug fix
- Fixed stub

## [2.0.1] - 2023-11-25

### Changed
- Improved stub

## [2.0.0] - 2023-11-24

### Changed
- Grouped critical components into `quent.pyx`
- Replaced `links` list with `next_link` linked structure
- Improved `isawaitable` performance
- Performance optimizations
- Use `__cinit__` for `Link`

### Removed
- `clone` method

## [1.9.1] - 2023-11-23

### Changed
- Minor improvements
- Stub fixes

## [1.9.0] - 2023-11-23

### Added
- Stub file

### Changed
- Miscellaneous fixes and improvements

## [1.8.4] - 2023-11-19

### Fixed
- Bug fix
- Minor fixes

## [1.8.3] - 2023-11-19

### Changed
- Performance improvements

## [1.8.2] - 2023-11-18

### Fixed
- `foreach` bug fix
- Bug fix

### Changed
- Structure changes

## [1.8.1] - 2023-11-18

### Fixed
- Bug fixes

## [1.8.0] - 2023-11-17

### Changed
- Split code into separate modules
- Updated `_handle_exception`

### Fixed
- Bug fix

## [1.7.0] - 2023-11-16

### Added
- Support for multiple `except` handlers

## [1.6.0] - 2023-11-16

### Changed
- Structure refactor


## [1.5.0] - 2023-11-15

### Added
- `clone` method
- `decorator` method

## [1.4.5] - 2023-11-13

### Fixed
- Re-raise uncaught exceptions if used `except_do`

## [1.4.4] - 2023-11-13

### Added
- `exceptions` parameter

## [1.4.3] - 2023-11-13

### Added
- `.freeze()` method

## [1.4.2] - 2023-11-13

### Changed
- Removed positional-only syntax for broader Python compatibility

## [1.4.1] - 2023-11-13

### Fixed
- Minor fixes

## [1.4.0] - 2023-11-13

### Added
- `.iterate()` method
- `.except_do()` method

### Changed
- Changed the behavior of `.foreach_do()`
- Use positional-arg syntax for public methods
- Improved `.with_()`
- Renamed `ignore` to `do`

## [1.3.1] - 2023-11-10

### Changed
- Set `autorun=False` by default

## [1.3.0] - 2023-11-10

### Added
- `autorun` configuration option

### Changed
- Do not `ensure_future` of nested `Chain`

### Fixed
- Miscellaneous fixes

## [1.2.0] - 2023-10-16

### Added
- Conditionals logic

## [1.1.0] - 2023-09-27

### Changed
- Schedule coroutines to run as tasks

### Fixed
- Fixed a bug with `.foreach_do()`

## [1.0.8] - 2023-09-27

### Fixed
- Fixed compiled Python type hints

## [1.0.7] - 2023-09-27

### Changed
- Set `embedsignature=True`
- Reorganized the main source

## [1.0.6] - 2023-09-27

### Changed
- Refactored `.foreach()` to be much faster

### Fixed
- Bug and test fixes

## [1.0.5] - 2023-09-26

### Added
- `.root_ignore()` method
- `async for` support for `foreach`

### Changed
- Major internal changes

### Removed
- `.async_with()` method

### Fixed
- Fixed a bug with `ignore_result`

## [1.0.4] - 2023-09-26

### Added
- `Chain.with_()` method

### Fixed
- Minor `with_` fix

## [1.0.3] - 2023-09-26

### Added
- `Chain.foreach()` method

## [1.0.2] - 2023-09-24

### Changed
- Renamed `pv` to `cv`
- Renamed `R` suffix to `Attr`
- Renamed `void` to `ignore`, `call` to `attr_fn`

## [1.0.1] - 2023-09-24

### Added
- `Chain.void()` method
- Tests and README

### Changed
- Formatted chain link exceptions
- Removed `eager` functionality

## [1.0.0] - 2023-09-23

### Added
- Initial release of quent
- Core `Chain` and `Cascade` classes
- Transparent async/sync handling
- Fluent API for operation chaining
