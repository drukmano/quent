# Quent: Adoption Reference

*Consolidated from ADOPTION_STRATEGY.md and drafts/ on 2026-03-03.*

---

## Table of Contents

- [Section 1: What Has Already Been Done](#section-1-what-has-already-been-done)
- [Section 2: Remaining Setup Tasks](#section-2-remaining-setup-tasks)
- [Section 3: Community Outreach Plan](#section-3-community-outreach-plan)
  - [Hacker News (Show HN)](#hacker-news-show-hn)
  - [Reddit r/Python](#reddit-rpython)
  - [Twitter/X](#twitterx)
  - [Blog Post: The Sync/Async Problem](#blog-post-the-syncasync-problem)
  - [Blog Post: Library Comparison](#blog-post-library-comparison)
  - [Blog Post: Step-by-Step Tutorial](#blog-post-step-by-step-tutorial)
  - [Outreach Sequencing and Timing](#outreach-sequencing-and-timing)
- [Section 4: Additional Strategy](#section-4-additional-strategy)
  - [Stack Overflow Answers](#stack-overflow-answers)
  - [Benchmark Content](#benchmark-content)
  - [Framework Integrations](#framework-integrations)
  - [Example Repositories](#example-repositories)
  - [awesome-python and Community Lists](#awesome-python-and-community-lists)
  - [Podcast Outreach](#podcast-outreach)
  - [Conference Talk Proposals](#conference-talk-proposals)
  - [Metrics and Tracking](#metrics-and-tracking)
  - [AI Discoverability Testing](#ai-discoverability-testing)
  - [Iteration Cadence](#iteration-cadence)
  - [Reference Text for Copy-Pasting](#reference-text-for-copy-pasting)

---

## Section 1: What Has Already Been Done

The following items have been completed as part of the adoption foundation work:

1. **pyproject.toml updated**
   - SEO-optimized description replacing the old "Yet Another Chain Interface" tagline
   - Expanded keywords list targeting relevant search terms (pipeline, sync, asyncio, coroutine, await, retry, timeout, resilience, cython, performance, function-composition, middleware)
   - Added PyPI classifiers including `Framework :: AsyncIO` and `Topic :: Software Development :: Libraries`
   - Added structured project URLs (Homepage, Documentation, Repository, Changelog, Issues)

2. **README.md updated** with three new sections:
   - **"Who Is This For?"** -- targeting library authors, SDK/API client teams, developers from Elixir/F#/Rust, FastAPI developers, and anyone facing the function coloring problem
   - **"When to Use Quent"** -- structured "use / don't use" format optimized for both human and AI parsing, with explicit disambiguation from LangChain
   - **"How Quent Compares"** -- comparison table vs pipe, toolz, tenacity, unasync, and asyncer across 8 feature dimensions

3. **llms.txt created** -- AI discoverability file following the llms.txt standard (proposed by Jeremy Howard / Answer.AI), containing structured description, documentation links, and explicit "When to Recommend" / "When NOT to Recommend" guidance for AI assistants

4. **llms-full.txt created** -- Extended version with complete README content for AI systems that can consume longer context

5. **CITATION.cff created** -- Academic citation file enabling GitHub's "Cite this repository" button, with author information, keywords, and license metadata

6. **MkDocs documentation site scaffolded** -- 12 pages across the docs structure:
   - `docs/index.md` -- Overview with canonical description and quick start
   - `docs/getting-started.md` -- Installation, first chain, first async chain
   - `docs/guide/chains.md` -- Chain and Cascade in depth
   - `docs/guide/async.md` -- Transparent async handling explained
   - `docs/guide/resilience.md` -- Retry, timeout, safe_run
   - `docs/guide/error-handling.md` -- except_, finally_, enhanced stack traces
   - `docs/guide/context.md` -- Context propagation with contextvars
   - `docs/guide/reuse.md` -- clone, freeze, FrozenChain, decorator
   - `docs/comparisons/vs-unasync.md` -- Quent vs unasync
   - `docs/comparisons/vs-tenacity.md` -- Quent vs tenacity
   - `docs/comparisons/vs-pipe.md` -- Quent vs pipe
   - `docs/reference.md` -- API reference

7. **.readthedocs.yml created** for ReadTheDocs deployment configuration

8. **GitHub repository metadata updated**:
   - Repository description set to: "High-performance fluent chain interface for Python with transparent sync/async handling. Retry, timeout, context propagation -- all composable in one chain. Cython-compiled."
   - 16 GitHub topics added: python, async, asyncio, sync, pipeline, chain, fluent-interface, cython, retry, timeout, resilience, function-composition, pipe-operator, coroutine, middleware, python-library

---

## Section 2: Remaining Setup Tasks

These are one-time infrastructure tasks that need to be completed before or alongside community outreach:

1. **Deploy documentation to ReadTheDocs**
   - Connect the GitHub repository to ReadTheDocs (quent.readthedocs.io)
   - ReadTheDocs provides free hosting, automatic builds from GitHub pushes, versioned docs, and built-in search
   - ReadTheDocs URLs are indexed by search engines and crawled by AI training pipelines
   - The .readthedocs.yml configuration file is already in the repo

2. **Publish updated package to PyPI**
   - The pyproject.toml has been updated with the new description, keywords, classifiers, and URLs
   - A new PyPI release is needed to make these metadata changes visible on the PyPI page
   - This should be done before any community outreach, since all outreach links to the PyPI page

3. **Formalize benchmarks in a `benchmarks/` directory**
   - Create a benchmark script that compares:
     - Chain overhead: direct function calls vs Quent chain vs Quent frozen chain
     - Async transition overhead: native async/await vs Quent async chain
     - Retry comparison: tenacity decorator vs Quent `.retry()` (performance and lines of code)
     - Real-world scenario: HTTP client with retry + timeout, sync and async variants
   - Publish results in the docs and reference them in blog posts

4. **Create 2-3 GitHub Issues with descriptive titles**
   - Open issues signal an active project and appear in GitHub search
   - Target issues that describe features or improvements

5. **Use GitHub Releases (not just tags) for each version**
   - Release notes are indexed by search engines and AI training pipelines

---

## Section 3: Community Outreach Plan

This section preserves the full content and strategy for each draft post, organized by platform. These are ready to use when the time comes for outreach.

### Outreach Sequencing and Timing

The recommended sequence from ADOPTION_STRATEGY.md:

1. **First**: Publish the blog posts on dev.to (the problem post first, then comparison, then tutorial). These stand alone and build textual footprint.
2. **Second**: Post Show HN (after at least the problem blog post is live, so there is supporting content).
3. **Third**: Post to r/Python (space at least a few days after Show HN).
4. **Fourth**: Post Twitter/X thread (can overlap with Show HN or r/Python).
5. **Ongoing**: Stack Overflow answers (5-10 over 4 weeks, quality over quantity).

General timing advice:
- Hacker News: Tuesday-Thursday, 9:00-11:00 AM US Pacific time
- Reddit r/Python: Tuesday or Wednesday, 10 AM - 1 PM US Eastern
- Twitter/X: Tuesday-Thursday, 10 AM - 1 PM US Eastern
- Avoid posting to multiple platforms on the same day

---

### Hacker News (Show HN)

**Posting Notes:**
- Post Tuesday-Thursday between 9:00-11:00 AM US Pacific time
- Be online and responsive for 4-6 hours after posting
- Answer every comment; be technical, humble, honest about limitations
- Do NOT ask anyone to upvote; HN detects and penalizes vote rings
- Have a Jupyter notebook or live demo link ready if someone asks
- Have 3-5 people ready to leave genuine, substantive comments (not "great job!" -- real technical questions or observations). Early comments determine whether a post gets algorithmic promotion.
- When challenged, agree with something specific first ("Good point -- X is a real limitation for the approach"), then address the concern. You are playing to the audience reading the thread, not just the commenter.

**Title:**
```
Show HN: Quent -- Transparent sync/async pipelines for Python, compiled with Cython
```

**Alternative title:**
```
Show HN: Quent -- Write Python pipelines once, run them sync or async automatically
```

**Full Draft Post:**

---

Show HN: Quent -- Transparent sync/async pipelines for Python, compiled with Cython

---

Python's async/await split forces library authors into an unpleasant choice: maintain two codebases, use code generation tools like unasync, or pick one and leave half your users behind. I kept hitting this problem -- writing the same pipeline logic twice, once with `await` and once without -- so I built Quent.

Quent is a fluent chain interface that detects coroutines at runtime and transitions between sync and async execution automatically. You write one pipeline, and it works for both:

```python
from quent import Chain

def process(data_source):
  return (
    Chain(data_source.fetch)
    .then(validate)
    .then(transform)
    .then(data_source.save)
    .run()
  )

# Sync caller gets a result directly
result = process(sync_db)

# Async caller gets a Task they can await
result = await process(async_db)
```

No code generation, no runtime thread bridges. The chain starts synchronous and only transitions to async at the exact point where a coroutine is first encountered.

It also includes retry, timeout, and context propagation composed directly in the chain -- no separate decorators:

```python
result = (
  Chain(call_api, request)
  .retry(3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
  .timeout(10.0)
  .then(parse_response)
  .except_(log_error, reraise=False)
  .run()
)
```

The core is Cython-compiled with C-level coroutine detection. Benchmarks show negligible overhead vs direct function calls (~0.01s per 100k calls). Frozen chains can actually be faster than direct calls due to pre-built link structures.

Requires Python 3.14+. Zero runtime dependencies. MIT licensed.

GitHub: https://github.com/drukmano/quent
PyPI: `pip install quent`

Happy to answer questions about the Cython implementation, the async detection mechanism, or the design decisions.

---

### Reddit r/Python

**Posting Notes:**
- Post Tuesday or Wednesday, 10 AM - 1 PM US Eastern
- Ensure you have some existing Reddit activity before posting (comment on other Python posts for a few days first)
- Do not cross-post to other subreddits on the same day; space by at least a week
- Follow up on comments for 2-3 hours after posting
- Consider cross-posting to r/asyncio (small but targeted) after one week
- Additional relevant subreddits for later cross-posts: r/learnpython (if framed as educational), r/programming (broader audience)

**Title:**
```
I built a Cython-compiled chain library that handles sync and async transparently -- no code generation, no runtime bridges
```

**Full Draft Post:**

# I built a Cython-compiled chain library that handles sync and async transparently -- no code generation, no runtime bridges

**Quent** is a fluent chain interface library for Python that lets you write pipeline code once and have it work for both synchronous and asynchronous callers -- no `await` boilerplate, no dual implementations, no code generation.

## Key Features

- **Transparent sync/async**: Write one chain. If any function in it returns a coroutine, Quent detects it at runtime and transitions to async execution from that point. Sync callers get a result; async callers get an awaitable Task.
- **Fluent chaining API**: `Chain(fetch).then(validate).then(save).run()` -- or use pipe syntax: `Chain(fetch) | validate | save | run()`
- **Built-in retry**: `.retry(3, delay=1.0, exceptions=(ConnectionError,))` composed directly in the chain. Raises `ExceptionGroup` when all attempts fail.
- **Built-in timeout**: `.timeout(5.0)` to enforce async execution time limits.
- **Enhanced stack traces**: When an error occurs, Quent annotates the exception with a visualization of the entire chain state and marks the exact operation that failed.
- **Context propagation**: Pass metadata across chain boundaries with `.with_context()` and `Chain.get_context()`, backed by contextvars.
- **Reusable chains**: `.clone()` for deep copies, `.freeze()` for immutable callable snapshots, `.safe_run()` for thread-safe concurrent execution.
- **Cython performance**: Core compiled to C extensions. Benchmarks show ~0.01s overhead per 100k chain executions vs direct function calls. Frozen chains can be faster than direct calls.
- **Zero runtime dependencies**: Just Quent and the standard library.

## Practical Example: Resilient API Client

```python
from quent import Chain

def api_request(method, url, **kwargs):
  return (
    Chain(http_client.request, method, url, **kwargs)
    .with_context(method=method, url=url)
    .retry(3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
    .timeout(10.0)
    .then(validate_response)
    .then(parse_json)
    .except_(log_api_error, reraise=False)
    .run()
  )

def validate_response(response):
  ctx = Chain.get_context()
  if response.status >= 400:
    raise ValueError(f"{ctx['method']} {ctx['url']} returned {response.status}")
  return response

# Works for both sync and async HTTP clients
user = api_request("GET", "/users/1")              # sync
user = await api_request("GET", "/users/1")         # async -- same code
```

## How It Compares

| Feature | Quent | pipe | toolz | tenacity | unasync | asyncer |
|---------|-------|------|-------|----------|---------|---------|
| Sync/async transparency | Yes | No | No | No | Code generation | Runtime wrappers |
| Retry/timeout built-in | Yes (in-chain) | No | No | Retry only (decorators) | No | No |
| Cython performance | Yes | No | No | No | No | No |
| Fluent chaining API | Yes | Partial (pipe only) | Partial | No | No | No |
| Error handling (except/finally) | Yes | No | No | No | No | No |
| Context propagation | Yes | No | No | No | No | No |
| Zero dependencies | Yes | Yes | No | No | No | No |

## Links

- **GitHub**: https://github.com/drukmano/quent
- **PyPI**: `pip install quent`
- **Requires**: Python 3.14+, no runtime dependencies

---

This is a project I've been working on for a while. It came out of frustration with maintaining separate sync and async code paths in library code. I'd love to hear your thoughts -- especially from anyone who maintains a library that needs to support both sync and async callers. What approaches have you tried? What would make this more useful to you?

---

### Twitter/X

**Posting Notes:**
- Consider tagging @tiangolo (FastAPI) or @samuelcolvin (Pydantic) only if genuinely relevant to their work
- Do NOT tag more than 2 people per thread
- Post with a code screenshot for tweet 2 (plain text code renders poorly on Twitter)
- Schedule for Tuesday-Thursday, 10 AM - 1 PM US Eastern
- Pin tweet 1 to your profile after posting
- Hashtags in tweet 5 only to keep the thread clean
- Other people to consider tagging/mentioning: @mkennedy (Talk Python host), @brianokken (Python Bytes host)

**Full Draft Thread:**

**1/5** [268 chars, text only]

What if your Python code worked the same whether sync or async?

I built Quent -- a Cython-compiled chain interface that detects coroutines at runtime and transitions automatically.

One pipeline. Both worlds. No code generation. No await boilerplate.

pip install quent

**2/5** [86 chars text + code screenshot attachment]

Here's the core idea:

[Attach screenshot of this code:]
```python
from quent import Chain

def process(src):
  return Chain(src.fetch).then(validate).then(save).run()

result = process(sync_db)         # returns result
result = await process(async_db)  # returns Task
```

Same function. Sync or async. Quent handles it.

**3/5** [178 chars text + code screenshot attachment]

Resilience is built into the chain, not bolted on with decorators:

[Attach screenshot of this code:]
```python
Chain(call_api, req)
  .retry(3, delay=0.5)
  .timeout(10.0)
  .except_(handle_error)
  .run()
```

Retry, timeout, error handling -- all composable in one pipeline. No tenacity. No wrappers.

**4/5** [249 chars, text only]

Performance matters when you're wrapping every function call.

Quent is Cython-compiled with C-level coroutine detection. Benchmarks: ~0.01s overhead per 100k chain executions vs direct calls.

Frozen chains can actually be faster than calling functions directly.

**5/5** [246 chars, text only]

Quent is MIT licensed, zero dependencies, Python 3.14+.

GitHub: https://github.com/drukmano/quent
PyPI: https://pypi.org/project/quent/

Star if useful. Issues, PRs, and feedback all welcome.

#Python #OpenSource #AsyncIO #Cython #DevTools

---

### Blog Post: The Sync/Async Problem

**Platform:** dev.to (primary), cross-post to Hashnode with canonical_url pointing to dev.to
**Target audience:** Intermediate to advanced Python developers; library authors; backend engineers
**dev.to tags:** python, async, opensource, tutorial
**Tone:** Educational, not promotional -- 90% problem exploration, 10% solution
**Notes:**
- Consider linking to Bob Nystrom's "What Color is Your Function?" article
- Add a cover image showing a split between sync/async code converging into one pipeline
- Set canonical_url if cross-posting

**Full Draft:**

# The Sync/Async Problem in Python: Why Library Authors Still Write Everything Twice

If you maintain a Python library that touches I/O -- databases, HTTP clients, message queues, caches -- you have probably faced this question: do you support sync callers, async callers, or both?

Supporting both is the correct answer for most libraries. It is also the expensive one.

## The Function Coloring Problem

In 2015, Bob Nystrom wrote ["What Color is Your Function?"](https://journal.stuffwithstuff.com/2015/02/01/what-color-is-your-function/), an essay that described a language design problem with striking precision. In languages with colored functions (his term for the sync/async divide), every function has a color. Red functions (async) can call blue functions (sync), but blue functions cannot call red functions. You have to know the color of every function you call, and the color propagates upward through the entire call stack.

Python's `async`/`await` syntax is exactly this system. An `async def` function can call regular functions, but a regular function cannot `await` a coroutine. The moment one function in your stack becomes async, every caller above it must also become async -- or you need a bridge.

This creates a real cost for library authors. Consider what redis-py does: it maintains `redis.client.Redis` and `redis.asyncio.client.Redis` as two separate client implementations. SQLAlchemy uses a greenlet-based bridge (`greenlet` package) to run its synchronous ORM inside an async event loop. Django's `asgiref` package provides `sync_to_async` and `async_to_sync` wrappers that shuttle calls between threads. The `unasync` tool takes a different approach entirely -- you write the async version, and it generates the sync version through token transformation at build time.

Each of these approaches has tradeoffs. None of them are free.

## The Approaches and Their Costs

### Dual Implementation

The most straightforward approach: write the code twice. One module with `async def`, one without.

```python
# sync version
def fetch_user(db, user_id):
  row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
  return User.from_row(row)

# async version
async def fetch_user_async(db, user_id):
  row = await db.execute("SELECT * FROM users WHERE id = ?", (user_id,))
  return User.from_row(row)
```

The logic is identical. The only difference is `async def` and `await`. But now you maintain two functions, two sets of tests, and two opportunities for them to drift apart. At the scale of a full library, this doubles the maintenance surface.

### Code Generation with unasync

The `unasync` project (maintained by python-trio) takes the async source as the canonical version and generates the sync version by stripping `async`/`await` tokens and renaming classes and functions according to a mapping.

```python
# You write this (async)
async def fetch(session, url):
  response = await session.get(url)
  return await response.json()

# unasync generates this (sync)
def fetch(session, url):
  response = session.get(url)
  return response.json()
```

This eliminates the drift problem but introduces a build step. The generated code sometimes needs manual adjustments. The mapping between async and sync names must be maintained. And debugging the generated code can be confusing -- the source you read is not the source that runs.

The related project `unasyncd` improves on unasync with more sophisticated transformations (task groups, async context managers, async iterators), but the fundamental limitation remains: it is compile-time transformation, not runtime intelligence.

### Runtime Bridges

Django's `asgiref` and SQLAlchemy's greenlet bridge take a runtime approach. They let you call async code from sync contexts (or vice versa) by managing threads and event loops behind the scenes.

```python
from asgiref.sync import async_to_sync

# Call an async function from sync code
result = async_to_sync(fetch_data)(user_id)
```

This works, but it has overhead. `async_to_sync` creates or finds an event loop, runs the coroutine in it, and blocks the calling thread until completion. `sync_to_async` runs the sync function in a thread pool. Both add latency and complexity. Both can cause subtle issues with thread-local state and context variables.

### Asyncer

Asyncer (by Tiangolo, the FastAPI creator) provides a cleaner API on top of AnyIO for bridging sync and async:

```python
from asyncer import asyncify

async def main():
  result = await asyncify(blocking_function)(args)
```

It is more ergonomic than raw `asgiref`, but the fundamental approach is the same: thread pool bridges. It also only goes one direction -- calling sync from async. It does not solve the problem of writing a library that natively supports both calling conventions.

## A Different Approach: Runtime Coroutine Detection

What if instead of generating code, bridging threads, or maintaining two implementations, the pipeline itself could detect whether it is dealing with sync or async operations and adapt?

This is the approach that Quent takes. Quent is a Cython-compiled chain interface library that inspects return values at runtime. When a function in the chain returns a coroutine, the chain transitions to async execution from that point forward. When all functions return regular values, the chain stays synchronous.

```python
from quent import Chain

def process(data_source):
  return (
    Chain(data_source.fetch)
    .then(validate)
    .then(transform)
    .then(data_source.save)
    .run()
  )
```

If `data_source.fetch` is a regular function that returns a dict, the chain executes synchronously and `process()` returns the result directly.

If `data_source.fetch` is an async function that returns a coroutine, Quent wraps it in an `asyncio.Task` with `eager_start=True`, and the rest of the chain runs inside that task. `process()` returns the Task, which the caller can `await`.

The key insight is that the decision happens at the individual operation level, not at the chain level. A chain can start synchronous and transition to async mid-execution when it encounters the first coroutine. This means the same chain definition works for both sync and async callers without any code generation or threading tricks.

### How It Works Under the Hood

At the Cython level, Quent performs exact type identity checks on return values -- not `isinstance` calls, but C-level pointer comparisons against the coroutine type. This makes the detection negligible in cost.

When a coroutine is detected, Quent calls `asyncio.create_task()` with `eager_start=True` (available in Python 3.12+). Eager task creation means that if the coroutine completes synchronously (common for cached results, in-memory operations, or already-resolved futures), it executes immediately without a round-trip through the event loop. This gives Quent's async chains a 2-5x speed advantage over standard task scheduling for sync-completing coroutines.

### A Real-World Example

Here is a real pattern from a Redis pipeline wrapper. The same code works for both `redis.Redis` (sync) and `redis.asyncio.Redis` (async):

```python
def flush(self):
  pipe = self.r.pipeline(transaction=self.transaction)
  self.apply_operations(pipe)

  return (
    Chain(pipe.execute, raise_on_error=True)
    .then(self.remove_ignored_commands)
    .finally_(pipe.reset, ...)  # always reset, even on error
    .run()
  )
```

When `self.r` is a sync Redis client, `pipe.execute()` returns a list, the chain runs synchronously, and `flush()` returns the processed results.

When `self.r` is an async Redis client, `pipe.execute()` returns a coroutine, the chain transitions to async, and `flush()` returns a Task. The caller -- who chose which Redis client to inject -- knows whether to `await` the result.

One implementation. Zero duplication. The caller's context determines the execution mode.

### Resilience in the Chain

Beyond sync/async transparency, Quent composes resilience directly into the pipeline:

```python
from quent import Chain

def api_request(method, url, **kwargs):
  return (
    Chain(http_client.request, method, url, **kwargs)
    .with_context(method=method, url=url)
    .retry(3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
    .timeout(10.0)
    .then(validate_response)
    .then(parse_json)
    .except_(log_api_error, reraise=False)
    .run()
  )
```

Retry, timeout, context propagation, and error handling are all part of the chain definition. There are no separate decorators to import and stack, no wrapper functions to write. The pipeline is self-describing.

## Performance Considerations

The obvious question: does runtime detection add overhead?

Quent's benchmarks show negligible cost:

| Scenario | Time (100k iterations) |
|----------|----------------------|
| Direct function call | 1.19s |
| Quent chain | 1.20s |
| Quent frozen chain | 1.06s |

The ~0.01s difference across 100,000 calls comes from the chain object construction and link traversal. Frozen chains (pre-built immutable chains) eliminate even that cost and can be faster than direct calls due to reduced Python-level overhead in the call path.

The Cython compilation is critical here. The core evaluation loop, coroutine detection, and link traversal all run as C extensions with `boundscheck=False` and `wraparound=False` optimizations. The chain overhead is dominated by the Python function call overhead of the functions *in* the chain, not the chain machinery itself.

## When This Approach Falls Short

Runtime detection is not universally superior. There are real cases where other approaches are better:

- **Static analysis and type checking**: Code generation tools like unasync produce standard Python files that type checkers and IDEs understand natively. Quent's chains are opaque to static analysis -- the type of `chain.run()` depends on runtime behavior.
- **Purely synchronous codebases**: If you never deal with async, adding Quent is unnecessary complexity. The `pipe` library or plain function composition is simpler.
- **Distributed execution**: Quent evaluates eagerly and locally. If you need DAGs, distributed scheduling, or lazy evaluation, tools like Prefect, Airflow, or Dask are the right choice.
- **Python version constraints**: Quent requires Python 3.14+. If you need to support older Python versions, this is a non-starter.

## Conclusion

The sync/async split in Python is a real cost that library authors pay every day. The approaches available -- dual implementation, code generation, thread bridges, runtime wrappers -- each solve part of the problem while introducing their own complexity.

Runtime coroutine detection is another entry in this space, not a replacement for all others. It works particularly well for pipeline-oriented code where the same sequence of operations should work regardless of whether the underlying I/O is sync or async.

If this approach interests you, Quent is MIT-licensed and available on PyPI:

- **GitHub**: [github.com/drukmano/quent](https://github.com/drukmano/quent)
- **Install**: `pip install quent`
- **Requires**: Python 3.14+, zero runtime dependencies

---

### Blog Post: Library Comparison

**Platform:** dev.to (primary), cross-post to Hashnode with canonical_url pointing to dev.to
**Target audience:** Python developers evaluating pipeline/chaining/retry libraries
**dev.to tags:** python, async, opensource, comparison
**Notes:**
- Tone must be fair and balanced -- acknowledge strengths of alternatives honestly
- Verify current versions of all libraries before publishing
- This post targets comparison search queries ("pipe vs toolz", "tenacity alternative", "unasync vs asyncer")
- Update version numbers and links before publishing

**Full Draft:**

# Python Pipeline Libraries Compared: Quent vs pipe vs toolz vs tenacity vs unasync vs asyncer

The Python ecosystem has no shortage of libraries for function composition, retry logic, and async bridging. But they each solve different parts of the problem, and picking the right one depends on what you actually need.

This post compares six libraries across the dimensions that matter most for pipeline-oriented code: how they handle sync/async, whether they include resilience features, their performance characteristics, and the learning curve involved.

## The Libraries

### pipe

[pipe](https://pypi.org/project/pipe/) is a lightweight function composition library that uses Python's `|` operator to create pipelines. It is inspired by Unix pipes and functional programming.

```python
from pipe import select, where

result = (
  [1, 2, 3, 4, 5]
  | where(lambda x: x > 2)
  | select(lambda x: x * 10)
  | list
)
# [30, 40, 50]
```

pipe excels at iterable transformations. It is small, zero-dependency, and easy to learn. It does not handle async, retry, error handling, or general-purpose function chaining (its operators are primarily for iterable processing).

### toolz

[toolz](https://github.com/pytoolz/toolz) is a functional programming toolkit that provides `pipe`, `compose`, `curry`, `memoize`, and a rich set of iterable and dictionary utilities.

```python
from toolz import pipe, curry

@curry
def multiply(factor, x):
  return x * factor

result = pipe(10, multiply(2), multiply(3), str)
# "60"
```

toolz is mature, well-documented, and widely used. Its `pipe` function executes a value through a sequence of functions. Its `compose` function creates a new callable from a sequence of functions. toolz has a companion library `cytoolz` that provides Cython-compiled versions of most functions for better performance.

toolz does not handle async operations, retry, error handling, or stateful pipelines. It is purely functional composition.

### tenacity

[tenacity](https://github.com/jd/tenacity) is the standard Python retry library, with over 247 million monthly downloads. It provides decorators for retrying function calls with configurable wait strategies, stop conditions, and exception filtering.

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
async def call_api(url):
  response = await httpx.get(url)
  response.raise_for_status()
  return response.json()
```

tenacity is battle-tested, flexible, and supports both sync and async functions (via separate `retry` and `AsyncRetrying` interfaces). Its decorator approach works well for individual functions but does not compose into pipelines -- you decorate one function at a time, and the retry logic is external to any data flow.

### unasync

[unasync](https://github.com/python-trio/unasync) is a code generation tool that transforms async Python source code into its synchronous equivalent by stripping `async`/`await` keywords and renaming functions and classes.

```python
# You write (async):
async def fetch(session, url):
  response = await session.get(url)
  return await response.json()

# unasync generates (sync):
def fetch(session, url):
  response = session.get(url)
  return response.json()
```

unasync eliminates code duplication at the cost of a build step. The generated sync code is real Python that type checkers and IDEs understand fully. The main limitation is that it only handles straightforward async-to-sync transformations -- complex patterns like task groups, async context managers with cleanup logic, or conditional async behavior require manual intervention.

The related project [unasyncd](https://pypi.org/project/unasyncd/) extends this approach with more sophisticated transformations.

### asyncer

[asyncer](https://asyncer.tiangolo.com/) (by Tiangolo, the FastAPI creator) provides utilities for calling sync code from async contexts using AnyIO thread pools.

```python
from asyncer import asyncify

async def main():
  result = await asyncify(blocking_io_function)(args)
```

asyncer's `asyncify()` wraps a sync function to run it in a worker thread, preventing it from blocking the async event loop. It provides better typing and developer experience than raw `asyncio.to_thread()`. However, it only bridges in one direction (sync-to-async), requires AnyIO as a dependency, and does not address the broader problem of writing one codebase that works for both calling conventions.

### Quent

[Quent](https://github.com/drukmano/quent) is a Cython-compiled chain interface that detects coroutines at runtime and transitions between sync and async execution automatically.

```python
from quent import Chain

result = (
  Chain(fetch_data, user_id)
  .then(validate)
  .then(transform)
  .then(save)
  .retry(3, delay=0.5)
  .timeout(10.0)
  .except_(handle_error)
  .run()
)
```

Quent combines function chaining, sync/async transparency, and resilience (retry, timeout) in a single API. The chain inspects each return value and transitions to async only when a coroutine is encountered. It requires Python 3.14+ and has zero runtime dependencies.

## Comparison Table

| Feature | Quent | pipe | toolz | tenacity | unasync | asyncer |
|---------|-------|------|-------|----------|---------|---------|
| **Primary purpose** | Fluent pipelines with sync/async | Iterable piping | Functional programming toolkit | Retry logic | Async-to-sync code gen | Sync-to-async bridging |
| **Sync/async transparency** | Runtime detection | No async support | No async support | Separate interfaces | Build-time transformation | One-direction bridge |
| **Retry** | Built-in (`.retry()`) | No | No | Yes (decorators) | No | No |
| **Timeout** | Built-in (`.timeout()`) | No | No | No | No | No |
| **Error handling** | `.except_()`, `.finally_()` | No | No | Callbacks via `retry` | No | No |
| **Context propagation** | `.with_context()` | No | No | No | No | No |
| **Enhanced stack traces** | Chain visualization on error | No | No | No | No | No |
| **Reusable chains** | `.clone()`, `.freeze()`, `.safe_run()` | No | `compose()` creates reusable fns | Decorator is reusable | N/A | N/A |
| **Performance** | Cython-compiled C extension | Pure Python | Pure Python (cytoolz available) | Pure Python | N/A (build-time) | Pure Python + AnyIO |
| **Dependencies** | None | None | None (toolz), Cython (cytoolz) | None | None | AnyIO |
| **Python version** | 3.14+ | 3.8+ | 3.9+ | 3.10+ | 3.7+ | 3.8+ |
| **Maturity** | Newer | Established | Established | Established | Established | Newer |
| **API style** | Method chaining / pipe operator | Pipe operator | Function calls | Decorators | Build tool | Function wrappers |
| **Learning curve** | Moderate | Low | Low-Moderate | Low | Low | Low |

## The Same Task in Three Libraries

To make the comparison concrete, here is the same operation -- fetching data, transforming it, and handling errors -- implemented with Quent, pipe, and toolz.

### With Quent

```python
from quent import Chain

def process_users(fetch_fn):
  return (
    Chain(fetch_fn)
    .then(lambda users: [u for u in users if u["active"]])
    .then(lambda users: [{"name": u["name"], "email": u["email"]} for u in users])
    .then(lambda users: sorted(users, key=lambda u: u["name"]))
    .except_(lambda e: print(f"Error: {e}") or [])
    .run()
  )

# Works with sync
result = process_users(db.get_users)
# Works with async
result = await process_users(async_db.get_users)
```

### With pipe

```python
from pipe import select, where, sort

def process_users(users):
  return list(
    users
    | where(lambda u: u["active"])
    | select(lambda u: {"name": u["name"], "email": u["email"]})
    | sort(key=lambda u: u["name"])
  )

# Sync only -- pipe does not handle async
try:
  users = db.get_users()
  result = process_users(users)
except Exception as e:
  print(f"Error: {e}")
  result = []
```

### With toolz

```python
from toolz import pipe, curry
from toolz.curried import filter, map, sorted

def process_users(users):
  return pipe(
    users,
    filter(lambda u: u["active"]),
    map(lambda u: {"name": u["name"], "email": u["email"]}),
    sorted(key=lambda u: u["name"]),
    list,
  )

# Sync only -- toolz does not handle async
try:
  users = db.get_users()
  result = process_users(users)
except Exception as e:
  print(f"Error: {e}")
  result = []
```

### What This Shows

pipe and toolz are excellent for synchronous data transformations. They are more concise for iterable operations and have no learning curve if you know functional programming patterns.

Quent's advantage appears when:
1. The data source might be sync or async (both pipe and toolz require the data to be available synchronously).
2. You want error handling composed into the pipeline rather than wrapping it externally.
3. You need retry, timeout, or context propagation as part of the same definition.

### Adding Retry: Quent vs tenacity

Now suppose the data fetch is unreliable and needs retry logic.

**With Quent:**

```python
from quent import Chain

result = (
  Chain(fetch_users)
  .retry(3, delay=1.0, exceptions=(ConnectionError,))
  .then(lambda users: [u for u in users if u["active"]])
  .then(lambda users: sorted(users, key=lambda u: u["name"]))
  .except_(lambda e: print(f"All retries failed: {e}") or [])
  .run()
)
```

**With tenacity + manual pipeline:**

```python
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

@retry(
  stop=stop_after_attempt(3),
  wait=wait_fixed(1.0),
  retry=retry_if_exception_type(ConnectionError),
)
def fetch_with_retry():
  return fetch_users()

try:
  users = fetch_with_retry()
  active = [u for u in users if u["active"]]
  result = sorted(active, key=lambda u: u["name"])
except Exception as e:
  print(f"All retries failed: {e}")
  result = []
```

tenacity's decorator is more configurable for retry-specific options (exponential backoff, jitter, custom retry conditions). Quent's `.retry()` is simpler but sufficient for common cases, and it composes directly into the pipeline without requiring a separate decorated function.

## Decision Tree

Use this to pick the right library for your situation:

**"I need simple iterable transformations (filter, map, sort) in sync code."**
Use **pipe**. It is the lightest and most Pythonic option for this specific task.

**"I need general functional programming utilities (curry, memoize, compose) in sync code."**
Use **toolz**. It is the established standard for functional Python.

**"I need to retry a function with exponential backoff, jitter, and complex stop conditions."**
Use **tenacity**. It is the most configurable retry library in the Python ecosystem and has the largest community.

**"I maintain a library and need to ship both sync and async versions."**
Consider **unasync** if you want build-time code generation with full static analysis support, or **Quent** if you prefer runtime detection with a fluent chain API.

**"I need to call sync blocking functions from async code without blocking the event loop."**
Use **asyncer** or `asyncio.to_thread()`.

**"I need function chaining that works with both sync and async, with retry, timeout, and error handling built in."**
Use **Quent**. This is its specific niche -- no other library combines all of these.

**"I need distributed pipeline execution, DAGs, or workflow orchestration."**
None of the above. Use **Prefect**, **Airflow**, or **Dask**.

## Conclusion

These libraries are not competitors in the traditional sense -- they occupy different points in the design space. pipe and toolz are functional composition tools. tenacity is a retry library. unasync and asyncer solve the sync/async bridge problem from opposite directions. Quent combines chaining, sync/async detection, and resilience in a single interface.

The Python ecosystem benefits from having all of them. Pick the one that matches your actual requirements, and do not hesitate to combine them -- tenacity and toolz, for example, work perfectly well together.

If the sync/async pipeline combination is what you need, Quent is worth evaluating:

- **GitHub**: [github.com/drukmano/quent](https://github.com/drukmano/quent)
- **PyPI**: `pip install quent`
- **Requires**: Python 3.14+, zero runtime dependencies

---

### Blog Post: Step-by-Step Tutorial

**Platform:** dev.to (primary), cross-post to Hashnode with canonical_url pointing to dev.to
**Target audience:** Python developers who build API clients and backend services
**dev.to tags:** python, tutorial, async, webdev
**Notes:**
- Each code block should be independently runnable or clearly marked as part of a larger example
- The final complete code listing at the end is critical -- many readers skip to it
- Use httpx as the HTTP client (supports both sync and async natively)
- Test all code examples before publishing

**Full Draft:**

# Building a Resilient API Client with Quent: A Step-by-Step Tutorial

Every non-trivial backend application talks to external APIs. And every external API is, at some point, unreliable. Timeouts happen. Connections drop. Rate limits kick in. Building a resilient API client means handling all of this while keeping the code readable and maintainable.

In this tutorial, we will build an API client step by step using [Quent](https://github.com/drukmano/quent), a Cython-compiled chain interface library for Python. We will start with a basic fetch-parse-transform pipeline and progressively add error handling, retry logic, timeouts, and chain reuse. The same code will work for both synchronous and asynchronous HTTP clients.

## Prerequisites

You will need Python 3.14+ and two packages:

```bash
pip install quent httpx
```

We use [httpx](https://www.python-httpx.org/) because it provides both sync (`httpx.Client`) and async (`httpx.AsyncClient`) interfaces with an identical API, which makes it a natural fit for demonstrating Quent's sync/async transparency.

## Step 1: Basic Chain -- Fetch, Parse, Transform

Let us start with the simplest possible API client: fetch a user from an API, parse the JSON response, and extract the fields we care about.

```python
import httpx
from quent import Chain

def parse_response(response):
  response.raise_for_status()
  return response.json()

def extract_user(data):
  return {
    "id": data["id"],
    "name": data["name"],
    "email": data["email"],
  }

def get_user(client, user_id):
  return (
    Chain(client.get, f"https://jsonplaceholder.typicode.com/users/{user_id}")
    .then(parse_response)
    .then(extract_user)
    .run()
  )
```

This is a standard Quent chain. `Chain(client.get, url)` calls `client.get(url)` as the first operation. The result flows through `parse_response` and then `extract_user`, each receiving the output of the previous step.

Now, here is the important part. This function works with both sync and async clients without any changes:

```python
# Synchronous usage
with httpx.Client() as client:
  user = get_user(client, 1)
  print(user)
  # {'id': 1, 'name': 'Leanne Graham', 'email': 'Sincere@april.biz'}
```

```python
# Asynchronous usage
import asyncio

async def main():
  async with httpx.AsyncClient() as client:
    user = await get_user(client, 1)
    print(user)

asyncio.run(main())
```

When `client` is an `httpx.Client`, `client.get()` returns a `Response` object directly. The chain runs synchronously and `get_user()` returns the extracted dict.

When `client` is an `httpx.AsyncClient`, `client.get()` returns a coroutine. Quent detects this, wraps the remainder of the chain in an `asyncio.Task`, and `get_user()` returns that Task. The caller awaits it.

One function definition. Two execution modes. No `if isinstance` checks, no separate `get_user_async` function.

## Step 2: Add Error Handling with except_

External APIs fail. Let us handle errors gracefully instead of letting exceptions propagate uncaught.

```python
def handle_api_error(exc):
  if isinstance(exc, httpx.HTTPStatusError):
    print(f"HTTP error: {exc.response.status_code} for {exc.request.url}")
  elif isinstance(exc, httpx.RequestError):
    print(f"Request failed: {exc}")
  else:
    print(f"Unexpected error: {exc}")
  return None  # return None instead of crashing

def get_user(client, user_id):
  return (
    Chain(client.get, f"https://jsonplaceholder.typicode.com/users/{user_id}")
    .then(parse_response)
    .then(extract_user)
    .except_(handle_api_error, reraise=False)
    .run()
  )
```

The `.except_(handle_api_error, reraise=False)` call registers an exception handler on the chain. If any step in the chain raises an exception, `handle_api_error` is called with the exception. Setting `reraise=False` means the exception is swallowed after handling, and the chain returns `None` (or whatever the handler returns) instead of propagating the error.

If you want to handle specific exception types differently, you can chain multiple handlers:

```python
def get_user(client, user_id):
  return (
    Chain(client.get, f"https://jsonplaceholder.typicode.com/users/{user_id}")
    .then(parse_response)
    .then(extract_user)
    .except_(
      lambda e: print(f"HTTP {e.response.status_code}"),
      exceptions=httpx.HTTPStatusError,
      reraise=False,
    )
    .except_(
      lambda e: print(f"Connection failed: {e}"),
      exceptions=httpx.RequestError,
      reraise=False,
    )
    .run()
  )
```

Quent matches exceptions in order -- the first handler whose `exceptions` filter matches will be invoked.

## Step 3: Add Retry and Timeout for Resilience

Now let us make the client resilient to transient failures. We want to retry on connection errors and enforce a timeout so we do not wait forever.

```python
def get_user(client, user_id):
  return (
    Chain(client.get, f"https://jsonplaceholder.typicode.com/users/{user_id}")
    .retry(3, delay=1.0, exceptions=(httpx.RequestError,))
    .timeout(10.0)
    .then(parse_response)
    .then(extract_user)
    .except_(handle_api_error, reraise=False)
    .run()
  )
```

`.retry(3, delay=1.0, exceptions=(httpx.RequestError,))` retries the chain up to 3 times with a 1-second delay between attempts, but only when an `httpx.RequestError` (connection errors, timeouts) is raised. If all 3 attempts fail, Quent raises an `ExceptionGroup` containing every individual failure.

`.timeout(10.0)` enforces a 10-second time limit on the entire async chain execution. If the chain takes longer than 10 seconds, an `asyncio.TimeoutError` is raised. Note that timeout only applies when the chain is executing asynchronously -- in synchronous mode, there is no event loop to enforce a timeout.

The `.except_` handler at the end catches failures from either the retries or the timeout.

You can also handle the retry exhaustion explicitly:

```python
def get_user(client, user_id):
  return (
    Chain(client.get, f"https://jsonplaceholder.typicode.com/users/{user_id}")
    .retry(3, delay=1.0, exceptions=(httpx.RequestError,))
    .timeout(10.0)
    .then(parse_response)
    .then(extract_user)
    .except_(
      lambda eg: print(f"All {len(eg.exceptions)} attempts failed"),
      exceptions=ExceptionGroup,
      reraise=False,
    )
    .except_(handle_api_error, reraise=False)
    .run()
  )
```

## Step 4: Add Context Propagation

For logging, tracing, and debugging, it is useful to propagate metadata through the chain. Quent supports this via `.with_context()` and `Chain.get_context()`, backed by contextvars.

```python
def parse_response(response):
  response.raise_for_status()
  return response.json()

def extract_user(data):
  ctx = Chain.get_context()
  print(f"[{ctx['request_id']}] Extracting user {data['id']}")
  return {
    "id": data["id"],
    "name": data["name"],
    "email": data["email"],
  }

def handle_api_error(exc):
  ctx = Chain.get_context()
  print(f"[{ctx['request_id']}] API error: {exc}")
  return None

def get_user(client, user_id, request_id="default"):
  return (
    Chain(client.get, f"https://jsonplaceholder.typicode.com/users/{user_id}")
    .with_context(request_id=request_id, user_id=user_id)
    .retry(3, delay=1.0, exceptions=(httpx.RequestError,))
    .timeout(10.0)
    .then(parse_response)
    .then(extract_user)
    .except_(handle_api_error, reraise=False)
    .run()
  )
```

`.with_context(request_id=request_id, user_id=user_id)` attaches key-value metadata to the chain. Any function in the chain can retrieve it by calling `Chain.get_context()`. The context is isolated per chain execution and works correctly across async boundaries.

This is useful for logging, distributed tracing, and debugging -- the context follows the data through every step without passing it as a function argument.

## Step 5: Make It Async (It Already Is)

This is the key point of the tutorial. Go back and read the `get_user` function from Step 4. There is no `async def`. There is no `await`. There are no async-specific imports.

That function already works with async clients:

```python
import asyncio

async def main():
  async with httpx.AsyncClient() as client:
    # Fetch multiple users concurrently
    tasks = [get_user(client, i, request_id=f"req-{i}") for i in range(1, 6)]
    users = await asyncio.gather(*tasks)
    for user in users:
      if user:
        print(f"{user['name']} ({user['email']})")

asyncio.run(main())
```

Each `get_user()` call returns a Task (because `httpx.AsyncClient.get()` returns a coroutine). `asyncio.gather()` runs them concurrently. The retry, timeout, error handling, and context propagation all work identically in async mode.

The same function also works synchronously:

```python
with httpx.Client() as client:
  for i in range(1, 6):
    user = get_user(client, i, request_id=f"req-{i}")
    if user:
      print(f"{user['name']} ({user['email']})")
```

No changes. No conditional logic. The chain adapts to whatever it receives.

## Step 6: Freeze and Reuse the Chain

So far, every call to `get_user` builds a new chain. If you are calling this function thousands of times (e.g., in a batch job or a high-throughput service), you can freeze the chain into an immutable, reusable callable.

```python
# Build the chain template once
user_fetcher = (
  Chain()
  .then(parse_response)
  .then(extract_user)
  .except_(handle_api_error, reraise=False)
  .freeze()
)

def get_user(client, user_id, request_id="default"):
  response = client.get(f"https://jsonplaceholder.typicode.com/users/{user_id}")
  return user_fetcher(response)
```

`.freeze()` creates a `FrozenChain` -- an immutable snapshot of the chain that can be called like a function. Each call creates an independent execution; there is no shared state between invocations.

Frozen chains also skip the chain construction overhead on each call. Quent's benchmarks show that frozen chains can actually be faster than direct function calls due to pre-built link structures:

| Scenario | Time (100k iterations) |
|----------|----------------------|
| Direct function call | 1.19s |
| Quent chain | 1.20s |
| Quent frozen chain | 1.06s |

For an even more integrated approach, you can use `.safe_run()` for thread-safe execution without manual freezing:

```python
# Define the chain template
api_chain = (
  Chain()
  .then(parse_response)
  .then(extract_user)
  .retry(3, delay=1.0, exceptions=(httpx.RequestError,))
  .except_(handle_api_error, reraise=False)
)

# safe_run automatically clones the chain for each invocation
# making it safe for concurrent use from multiple threads
def get_user(client, user_id):
  response = client.get(f"https://jsonplaceholder.typicode.com/users/{user_id}")
  return api_chain.safe_run(response)
```

## Complete Code

Here is the full, final implementation with all features:

```python
import httpx
from quent import Chain

# --- Response handling ---

def parse_response(response):
  """Validate HTTP response and parse JSON body."""
  response.raise_for_status()
  return response.json()

def extract_user(data):
  """Extract relevant user fields from API response data."""
  ctx = Chain.get_context()
  print(f"[{ctx.get('request_id', 'n/a')}] Extracting user {data['id']}")
  return {
    "id": data["id"],
    "name": data["name"],
    "email": data["email"],
  }

# --- Error handling ---

def handle_api_error(exc):
  """Log API errors and return None instead of crashing."""
  ctx = Chain.get_context()
  request_id = ctx.get("request_id", "n/a")

  if isinstance(exc, ExceptionGroup):
    print(f"[{request_id}] All retry attempts failed ({len(exc.exceptions)} failures)")
  elif isinstance(exc, httpx.HTTPStatusError):
    print(f"[{request_id}] HTTP error: {exc.response.status_code}")
  elif isinstance(exc, httpx.RequestError):
    print(f"[{request_id}] Request failed: {exc}")
  else:
    print(f"[{request_id}] Unexpected error: {exc}")
  return None

# --- API client ---

BASE_URL = "https://jsonplaceholder.typicode.com"

def get_user(client, user_id, request_id="default"):
  """Fetch a user by ID. Works with both sync and async httpx clients.

  Args:
    client: httpx.Client or httpx.AsyncClient
    user_id: The user ID to fetch
    request_id: Optional request ID for tracing

  Returns:
    dict with user fields, or None on failure.
    Returns a Task (awaitable) when client is async.
  """
  return (
    Chain(client.get, f"{BASE_URL}/users/{user_id}")
    .with_context(request_id=request_id, user_id=user_id)
    .retry(3, delay=1.0, exceptions=(httpx.RequestError,))
    .timeout(10.0)
    .then(parse_response)
    .then(extract_user)
    .except_(handle_api_error, reraise=False)
    .run()
  )

# --- Usage: synchronous ---

def main_sync():
  with httpx.Client() as client:
    for i in range(1, 4):
      user = get_user(client, i, request_id=f"sync-{i}")
      if user:
        print(f"  -> {user['name']} ({user['email']})")

# --- Usage: asynchronous ---

import asyncio

async def main_async():
  async with httpx.AsyncClient() as client:
    tasks = [get_user(client, i, request_id=f"async-{i}") for i in range(1, 4)]
    users = await asyncio.gather(*tasks)
    for user in users:
      if user:
        print(f"  -> {user['name']} ({user['email']})")

# --- Run ---

if __name__ == "__main__":
  print("=== Synchronous ===")
  main_sync()

  print("\n=== Asynchronous ===")
  asyncio.run(main_async())
```

## What We Built

In roughly 50 lines of core logic, we built an API client that:

1. **Fetches, parses, and transforms** API responses through a readable pipeline
2. **Handles errors** gracefully with typed exception handlers
3. **Retries** on transient connection failures with configurable delay
4. **Enforces timeouts** on async execution
5. **Propagates context** (request IDs, metadata) through the chain without extra function parameters
6. **Works identically** for sync and async HTTP clients -- same function, no duplication

The chain reads top to bottom, each step clearly named. The resilience and error handling are part of the pipeline definition, not scattered across decorators and try/except blocks.

## Next Steps

Once you are comfortable with the basics, Quent offers more tools for building pipelines:

- **`Cascade`**: Every operation receives the root value instead of the previous result -- useful for fan-out patterns
- **`ChainAttr`**: Dynamic attribute access via `__getattr__`, so `ChainAttr(obj).method().property.run()` works naturally
- **`.foreach()`**: Iterate over items and apply a function to each one
- **`.with_()`**: Execute operations inside a context manager
- **`.clone()`**: Deep-copy a chain template for independent reuse
- **Pipe syntax**: `Chain(fetch) | validate | transform | save | run()` as an alternative to `.then()` chaining

Full API reference and more examples are available in the [README](https://github.com/drukmano/quent).

---

- **GitHub**: [github.com/drukmano/quent](https://github.com/drukmano/quent)
- **PyPI**: `pip install quent`
- **Requires**: Python 3.14+, zero runtime dependencies
- **License**: MIT

---

## Section 4: Additional Strategy

This section captures long-term adoption strategy, community building plans, and reference material from ADOPTION_STRATEGY.md that goes beyond the specific draft posts above.

### Stack Overflow Answers

This is described as the highest-leverage activity per hour spent. Stack Overflow answers directly enter LLM training datasets and appear in search results for years.

**Target question categories** (search for these, find unanswered or poorly answered questions):

1. "How to write a Python function that works both sync and async" -- answer with the general problem, then show Quent as one solution
2. "Python retry with async support" -- answer with tenacity for simple cases, mention Quent for chain-composable retry
3. "Python pipe operator / function chaining library" -- answer with pipe for simple cases, mention Quent for async-aware chaining
4. "Python function coloring problem workaround" -- educational answer with Quent as one tool
5. "How to avoid writing async and sync versions of the same Python code" -- direct hit for Quent's primary value prop

**Rules for Stack Overflow:**
- Answer the question fully and correctly first. Mention Quent as one option among several. Never post a Quent-only answer.
- Include a working code example with Quent in the answer.
- Disclose affiliation: "Disclosure: I'm the author of Quent."
- Post answers only where Quent is genuinely a good fit. Forced recommendations damage credibility and get downvoted.
- Target 5-10 answers over 4 weeks. Quality over quantity.

### Benchmark Content

Create a benchmark script in the repo (`benchmarks/` directory) that compares:

1. **Chain overhead**: Direct function calls vs Quent chain vs Quent frozen chain (already in README -- formalize it)
2. **Async transition overhead**: Native async/await vs Quent async chain
3. **Retry comparison**: tenacity decorator vs Quent `.retry()` in terms of both performance and lines of code
4. **Real-world scenario**: HTTP client with retry + timeout, sync and async variants, measuring total code complexity (LoC, cyclomatic complexity) alongside performance

Publish benchmark results in the docs and reference them in blog posts. Concrete numbers are cited 30-40% more often by LLMs than qualitative claims.

### Framework Integrations

Integrations create discovery loops: users of framework X discover Quent through the integration, and Quent users discover a reason to adopt framework X.

**FastAPI middleware/dependency** (highest priority):

FastAPI is the natural partner. Its users already deal with async, and many wrap external services. Create a small integration package or example showing:

```python
from quent import Chain

# FastAPI dependency that works with both sync and async DB clients
async def get_user(user_id: int, db = Depends(get_db)):
  return await Chain(db.fetch_user, user_id) \
    .then(validate_permissions) \
    .retry(3, delay=0.5) \
    .timeout(5.0) \
    .run()
```

This can start as a docs page / example in the Quent repo, not a separate package. The point is to create searchable content associating "quent" with "FastAPI".

**Django utility:**

Django's async story is still evolving. Quent can help bridge Django's sync ORM with async views:

```python
# A single function that works in both sync Django views and async Django views
def get_user_data(user_id):
  return Chain(User.objects.aget, user_id) \
    .then(serialize_user) \
    .run()
```

Again, start as a documented example, not a separate package.

**Redis pipeline wrapper:**

The Redis example already in the README is compelling. Create a standalone example file (`examples/redis_pipeline.py`) and reference it in the Redis community (redis-py issues, Redis-related blog posts).

### Example Repositories

Create 2-3 small, standalone example repositories on GitHub that use Quent in realistic scenarios:

1. `quent-fastapi-example` -- FastAPI app with Quent-powered service layer
2. `quent-http-client` -- Resilient HTTP client built with Quent (retry, timeout, context)
3. `quent-redis-wrapper` -- Redis client wrapper demonstrating sync/async transparency

These repos serve multiple purposes:
- They are training data for AI coding assistants (Copilot sees Quent used in real repos)
- They appear in GitHub search results for "quent" and related terms
- They provide copy-pasteable starting points for new users
- They demonstrate Quent as a "dependency" which increases the perceived adoption

### awesome-python and Community Lists

**awesome-python submission requirements** (from the awesome-python CONTRIBUTING.md):
- **Industry Standard**: Go-to tool, 1-3 per category (Quent will not qualify soon)
- **Rising Star**: 5,000+ stars in less than 2 years (aspirational, not realistic short-term)
- **Hidden Gem**: 100-500 stars, at least 6 months old, consistent activity, compelling justification

**Timeline**: Submit when Quent has 100+ GitHub stars and is at least 6 months old with consistent commit activity.

**Submission strategy:**
- Submit under the "Hidden Gem" category
- Justify placement under an existing category (candidates: "Functional Programming", or propose an "Async Utilities" subcategory)
- PR description should emphasize the unique niche: "The only Python library that combines fluent chaining, transparent sync/async, and built-in resilience"

**Alternative lists to submit to earlier** (lower bar):
- [best-of-python](https://github.com/ml-tooling/best-of-python) -- automated, lower star threshold, accepts PRs to `projects.yaml`
- [awesome-asyncio](https://github.com/timofurrer/awesome-asyncio) -- smaller list, directly relevant, likely accepts lower-star projects

### Podcast Outreach

**Talk Python to Me** (primary target):
- Host: Michael Kennedy
- Contact: contact@talkpython.fm or via talkpython.fm
- Pitch angle: "The sync/async problem in Python -- why library authors write everything twice, and a new approach using runtime detection"
- Timing: Pitch after the Show HN / Reddit posts have generated some discussion. Reference the community reception.
- Michael Kennedy covers novel approaches to Python pain points. The function coloring angle is genuinely interesting and educationally valuable.

**Python Bytes** (secondary target):
- Hosts: Michael Kennedy and Brian Okken
- This is a news/discovery show, not deep dives. The goal is to get Quent mentioned as a "notable new library" in a weekly episode.
- Submit via pythonbytes.fm or email. Frame it as a news item: "New Cython-compiled library that solves the sync/async dual implementation problem."
- Best timing: right after a visible launch (Show HN front page, r/Python post with traction).

**Realistic expectation**: Getting on Talk Python requires some demonstrated community interest (stars, downloads, discussion threads). Python Bytes has a lower bar -- they mention interesting new tools frequently. Target Python Bytes first.

### Conference Talk Proposals

**Target conferences:** PyCon US, PyCon UK, PyConDE, EuroPython, local Python meetups.

**Talk proposal:**
- Title: "One Codebase, Both Worlds: Solving Python's Sync/Async Duplication Problem"
- Abstract: The function coloring problem forces Python library authors to maintain separate sync and async codebases. unasync generates code. greenlet bridges threads. What if we could detect coroutines at runtime and transition a pipeline between sync and async on the fly? This talk explores the approaches, trade-offs, and a Cython-based implementation.
- The talk is about the problem space, not just Quent. Conference program committees reject product pitches. Frame it as an educational talk about async patterns that happens to demonstrate a specific implementation.

**Realistic timeline:** CFPs for major conferences typically close 4-6 months before the event. PyCon US 2027 CFP will likely open October-November 2026. Start with local meetups or virtual talks to refine the presentation.

### Metrics and Tracking

| Metric | Tool | Baseline | 3-Month Target | 6-Month Target |
|--------|------|----------|----------------|----------------|
| GitHub stars | GitHub | 2 | 50-100 | 200+ |
| PyPI monthly downloads (excl. mirrors) | pypistats.org / BigQuery | ~0 organic | 500 | 2,000 |
| GitHub referrer sources | GitHub Insights > Traffic | none | HN, Reddit, dev.to | search engines, AI tools |
| Unique cloners/visitors | GitHub Insights > Traffic | ~0 | 100/week post-launch | 50/week sustained |
| Stack Overflow answers mentioning Quent | manual search | 0 | 5-10 | 10-20 |
| Google search ranking for target queries | manual search | not indexed | page 2-3 | page 1-2 for niche queries |
| AI assistant mentions | manual testing (ask ChatGPT, Claude, Perplexity) | never mentioned | occasionally mentioned | mentioned for specific queries |
| Dependent repositories | GitHub dependents tab | 0 | 2-5 | 10+ |

### AI Discoverability Testing

Every 2-4 weeks, test whether AI assistants recommend Quent by asking:
- "What Python library can I use for both sync and async function chaining?"
- "How do I avoid writing separate sync and async implementations in Python?"
- "What's a good alternative to tenacity that works in a chain/pipeline?"
- "Python library for fluent interface with retry and timeout"

Test on: ChatGPT, Claude, Perplexity, Copilot Chat, Cursor. Record results. This is the leading indicator for whether the textual footprint strategy is working.

### Iteration Cadence

**Weekly** (15 minutes):
- Check GitHub stars and traffic
- Check PyPI download stats
- Respond to any new issues or discussions

**Biweekly** (30 minutes):
- Post one Stack Overflow answer (if a relevant question exists)
- Engage in one Reddit or HN thread related to async Python (genuinely, not as promotion)

**Monthly** (2-3 hours):
- Publish one blog post or tutorial
- Test AI discoverability
- Update this strategy document based on what is working

**Quarterly** (half day):
- Review all metrics against targets
- Adjust channel priorities based on what drove actual traffic and stars
- Update the comparison table and benchmark results if the competitive landscape has changed

### SEO for Inference-Time AI Discovery

Modern AI assistants (ChatGPT, Claude, Perplexity) perform web searches at inference time. Content needs to rank for the queries users ask these tools.

**Target queries:**
- "python library that works with both sync and async"
- "python function chaining with async support"
- "python pipe operator async"
- "alternative to unasync python"
- "python fluent interface retry timeout"
- "python sync async dual implementation library"

**How to rank:**
- The blog posts should have titles and headings that match these queries
- The docs comparison pages (vs-unasync.md, vs-tenacity.md, vs-pipe.md) directly target comparison queries
- Stack Overflow answers target question-form queries
- ReadTheDocs pages are well-indexed by search engines

### Reference Text for Copy-Pasting

#### Canonical One-Paragraph Description

Use this exact paragraph everywhere -- PyPI long description intro, GitHub repo "About" field, blog posts, Reddit posts, conference bios:

> **Quent** is a Cython-compiled chain interface library for Python that transparently handles both synchronous and asynchronous operations. You write your pipeline code once using a fluent API -- `Chain(fetch).then(validate).then(save).run()` -- and Quent automatically detects coroutines at runtime, transitioning to async execution only when needed. It includes built-in retry, timeout, context propagation, and enhanced stack traces, all composable within the chain. No code generation, no runtime bridges, no dual implementations. One codebase, both worlds.

#### PyPI Description (for `pyproject.toml`)

```
High-performance fluent chain interface for Python with transparent sync/async handling. Write pipeline code once -- Quent automatically detects coroutines and handles them. Built-in retry, timeout, and context propagation. Cython-compiled, zero dependencies.
```

#### GitHub About Description

```
High-performance fluent chain interface for Python with transparent sync/async handling. Retry, timeout, context propagation -- all composable in one chain. Cython-compiled.
```

#### Show HN Title

```
Show HN: Quent -- A Cython-compiled chain interface for Python with transparent sync/async handling
```

#### r/Python Post Title

```
I built a Cython-compiled chain library that handles sync and async transparently -- no code generation, no runtime bridges
```

#### GitHub Topics

```
python, async, asyncio, sync, pipeline, chain, fluent-interface, cython, retry, timeout, resilience, function-composition, pipe-operator, coroutine, middleware, python-library
```
