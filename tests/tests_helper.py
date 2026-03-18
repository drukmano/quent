"""Exhaustive sync/async bridge test infrastructure.

This module re-exports all public symbols from the split sub-modules for
backward compatibility.  New code should import from the individual modules
directly:

  - ``tests.fixtures``        -- sentinels, result capture, callable pairs, CMs, variant axes
  - ``tests.symmetric``       -- SymmetricTestCase base class
  - ``tests.bricks``          -- Brick definitions, ALL_BRICKS, REPRESENTATIVE_BRICKS
  - ``tests.handler_configs`` -- HandlerConfig, handler oracles, error injection types
  - ``tests.bridge_runner``   -- run_bridge(), oracle verification, gather comparison
"""

from __future__ import annotations

# --- bricks ---
from tests.bricks import (  # noqa: F401
  ALL_BRICKS,
  BRICKS_BY_OP,
  REPRESENTATIVE_BRICKS,
  Brick,
)

# --- bridge_runner ---
from tests.bridge_runner import (  # noqa: F401
  BridgeStats,
  compose_oracles,
  gather_results_equivalent,
  run_bridge,
  verify_all_brick_oracles,
  verify_brick_oracle,
)

# --- fixtures ---
from tests.fixtures import (  # noqa: F401
  V_ADD,
  V_BAD_CLEANUP,
  V_CM,
  V_CM_SUPPRESSES,
  V_DOUBLE,
  V_FALSE,
  V_FN,
  V_GT0,
  V_HANDLER,
  V_IDENTITY,
  V_IS_EVEN,
  V_IS_TRUTHY,
  V_ITER,
  V_KW,
  V_NOOP,
  V_RAISE,
  V_TRIPLE,
  V_TRUE,
  AsyncCM,
  AsyncCMSuppresses,
  AsyncPair,
  AsyncRange,
  CustomBaseError,
  CustomError,
  DualProtocolCM,
  Result,
  SyncCM,
  SyncCMAsyncExit,
  SyncCMSuppresses,
  VariantAxis,
  async_add,
  async_always_false,
  async_always_true,
  async_bad_cleanup,
  async_double,
  async_fn,
  async_gt0,
  async_handler,
  async_identity,
  async_is_even,
  async_is_truthy,
  async_kw,
  async_noop,
  async_raise,
  async_triple,
  capture,
  sync_add,
  sync_always_false,
  sync_always_true,
  sync_bad_cleanup,
  sync_double,
  sync_fn,
  sync_gt0,
  sync_handler,
  sync_identity,
  sync_is_even,
  sync_is_truthy,
  sync_kw,
  sync_noop,
  sync_raise,
  sync_triple,
)

# --- handler_configs ---
from tests.handler_configs import (  # noqa: F401
  ERROR_INJECTION_TYPES,
  HANDLER_CONFIGS,
  HandlerConfig,
  async_break_signal,
  async_except_consume,
  async_except_fails,
  async_except_kwargs,
  async_except_noop,
  async_finally_kwargs,
  async_finally_ok,
  async_raise_base,
  async_return_signal,
  sync_break_signal,
  sync_except_consume,
  sync_except_fails,
  sync_except_kwargs,
  sync_except_noop,
  sync_finally_kwargs,
  sync_finally_ok,
  sync_raise_base,
  sync_return_signal,
)

# --- symmetric ---
from tests.symmetric import SymmetricTestCase  # noqa: F401
