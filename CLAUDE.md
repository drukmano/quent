# CLAUDE.md

## Project Identity

quent is a **transparent sync/async bridge**. It is not a collections library, not a functional programming toolkit, not an opinionated framework. It provides the minimum set of pipeline primitives that let developers:

- **Write code once** — a single chain definition works for both sync and async callables. Zero ceremony, zero code duplication.
- **Migrate existing codebases** — unify separate sync and async implementations of the same logic into one. Stop maintaining two versions of every function.
- **Stay out of the way** — quent is unopinionated. It bridges the sync/async divide and gets out of the way. No imposed patterns, paradigms, or abstractions beyond the pipeline itself.

**What quent is NOT:** a collections/iteration library (use itertools, more-itertools), a functional programming framework (use returns, toolz, Expression), or an opinionated architecture (use Effect, returns).

**Design principle:** Every feature must justify itself by solving a real sync/async bridging problem or eliminating genuine code duplication. Features that merely wrap stdlib functionality don't belong.

