---
name: ommat
description: Use when turning LLM or decision-model answers into a checked, replayable plan with ommat, the Python decision engine; when writing or changing a design (sources, questions, rules, policy, steps, requirements); when calling load, decide or replay; when making agent decisions auditable or replayable with ommat; or when binding ommat to models through ommat[idia].
---

# ommat

ommat turns many small model answers into one checked plan. A model answers small questions; your rules and policy, plain Python functions, turn the answers into a plan; a guard checks it; and a record replays the turn with no model call.

This skill holds the pattern and the pitfalls. Every detail lives in the docs linked here; read them rather than guessing.

## Install

```sh
pip install "ommat @ git+https://github.com/CRSiegfried/ommat@v0.4.0"
```

The core imports only the standard library.

## The core pattern

A `Design` holds:

- `state`: the type of the turn's input, read as `t:state`.
- `sources`: each asks an oracle `Question`s (`noul`, `choice`, `score` or `extraction`) about its `needs`, its answers landing at `q:` refs, or provides typed `t:` values.
- `rules`: each a function from a read view to one `d:` estimate, with its `reads` declared.
- `policy`: one function that reads estimates and returns a `Plan`: `execute`, `clarify`, `inform` or `escalate`.
- `Step`s in an `execute` plan: a tool from `tools`, arguments, and the `Requirement`s that must hold.
- `requirements`: the design's own `Requirement`s, checked on every `execute` plan.
- bindings: async functions the caller passes to decide, keyed by oracle name (`default` for a source that names none).

```python
from ommat import decide_sync, load, replay_sync

loaded = load(DESIGN)  # raises InvalidDesign with every failure
decision = decide_sync(loaded, state, {"default": binding})
decision.plan, decision.verdict, decision.record
# Replay makes no model call; [] means no difference; rerun is None if nothing ran.
rerun, findings = replay_sync(decision.record, loaded)
```

A full, runnable example with its output is the [README demo](https://github.com/CRSiegfried/ommat/blob/v0.4.0/README.md#a-first-decision). Build a design step by step with the tutorial [Write your first design](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/tutorials/write-a-design.md).

## Rules of the engine

- An unknown is a marker, `Unsure`, `Unavailable` or `Unasked` (all `Undetermined`), never `None` or a quiet no. Test reads with `isinstance(value, Undetermined)` before using them, in rules and in the policy; `if value:` on a marker raises.
- `decide` calls no tool. The caller runs the steps of a plan whose verdict passed and whose kind is `execute`, on its own engine or with `ommat.helpers.run_plan`. Dispatch on `plan.kind`, never on the verdict alone: the guard passes a `broken` plan, which has no steps. See the [spec's Decide section](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#decide).
- Define the design's rules, `when`s and policy in an importable module, not in `__main__`, so a record replays under it in another process with no `design-mismatch`. The [spec's Replay section](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#replay) says how.

## Testing

- Bind a stand-in: an `async def binding(batch, *, seconds) -> CallResult` giving a fixed answer for each `q:` ref in each `SourceCall`'s `questions`, probabilities keyed by label name, an extraction's answer through `ommat.bindings.extraction_value`. The README demo has one. A binding without the `seconds` keyword fails every call.
- `ommat.bindings.conformance.check(binding)`, outside the compatibility promise, holds a binding of your own to a provider to the protocol: it returns the failures, `[]` on a pass, and must be called outside a running event loop.
- Assert on `decision.plan.kind`, `decision.verdict.kind` and the replay findings.
- Two design lints run the design with no model: `ommat.helpers.dead_questions(loaded)` reports, per question, whether its answer changed the plan or the verdict at any point it tried (`live`, `no-change-found` or `not-on-grid`), and `ommat.helpers.unsure_to_execute(loaded)` reports each point where the policy, given a `d:` read set to `Unsure`, returns an `execute` plan.

## Pitfalls

- **load refuses what 0.4.0 leaves out:** decision tables, data rules, plan declarations and candidate sets. The list is in the [spec's intro](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md).
- **Nothing carries between turns.** Each turn stands alone; the caller passes what a later turn needs, such as an earlier step's result, in its input state, in fields declared in `state`: decide drops an undeclared field without a word. See [Continuing a workflow](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#continuing-a-workflow).
- **Declare every read.** A rule that reads a ref outside its `reads` makes its `d:` ref `Unavailable` (`undeclared-read`); the policy's makes the plan `broken`. Catching it does not help.
- **A step with no requirement of its own refuses the plan**, unless the design sets `default_deny=False`. Give each step the `Requirement`s it rests on.
- **In async code, await `decide` and `replay`.** The `_sync` forms raise `RuntimeError` inside a running event loop, as in Jupyter.
- **Don't change a value you read** in a rule, a `when` or the policy; see the [spec's Decide section](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#decide).

## Using with idia

[idia](https://github.com/CRSiegfried/idia) puts ommat's questions to model providers. Install both from their tags, idia first with the extras for your providers:

```sh
pip install "idia[openai,cloudflare] @ git+https://github.com/CRSiegfried/idia@v0.1.0"
pip install "ommat[idia] @ git+https://github.com/CRSiegfried/ommat@v0.4.0"
```

- `ommat.bindings.idia.oracle_binding(dispatcher)` makes a binding from an idia `Dispatcher` over one leaf, with its re-ask off and its overflow policy `Fail`, as `idia.provider(name)` builds one. It refuses any other provider, the `cloudflare()` router in `idia.cloudflare` among them: bind Clef and Clairic as two oracles. Bind one per oracle; a design that asks different models names one oracle each.
- A binding receives each source's questions together, and `oracle_binding` sends them to the provider as one request, split only where the provider's limits require. Put questions about the same input in one source to batch them. A source asking an extraction sends only its text and images, and its extractions must read one text and one reference date; ask label questions about other needs in a source of their own.
- `oracle_binding` follows idia while idia is 0.x, so it may change in any minor release; it is outside ommat's compatibility promise.
- For idia itself (providers, credentials, errors), install idia's own skill from [skills/idia](https://github.com/CRSiegfried/idia/tree/v0.1.0/skills/idia).
- Walk through a real binding with the tutorial [Decide with real models through idia](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/tutorials/decide-with-idia.md).

## Details

- [The spec](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md): what ommat does, and what 0.4.0 leaves out.
- [Write your first design](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/tutorials/write-a-design.md) and [Decide with real models through idia](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/tutorials/decide-with-idia.md).
- [The README demo](https://github.com/CRSiegfried/ommat/blob/v0.4.0/README.md#a-first-decision).
- [The public API](https://github.com/CRSiegfried/ommat/blob/v0.4.0/CONTRIBUTING.md#ground-rules-for-changes): the names under the compatibility promise.
