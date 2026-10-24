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
- `sources`: each asks an oracle `Question`s (`noul`, `choice`, `score` or `extraction`) about its `needs`, its answers landing at `q:` refs, or provides typed `t:` values. A question is worded by a `Text` template, or a `Structured` JSON object or array whose string values are templates; a `noul` may describe its `yes` and `no`.
- `rules`: each keyed by its name and providing `d:<name>`, a function from a read view to one estimate, with its `reads` declared. Source ids, rule ids, parameter paths and tool names share one namespace, so a rule and a source cannot share a name.
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

## Beyond the basics

Three question shapes from format 4, as literals:

```python
from ommat import Question, Source, Structured, Text
from ommat import types as t

# A choice over candidates a source finds: none and several first, then the candidates.
finder = Source({"t:senders": {"type": t.list_(t.string), "candidates": True}}, needs=["t:state"])
sender = Question(
    "choice",
    Text("Who sent this message?"),
    {"fixed": ["none", "several"], "candidates": "t:senders", "at": 2},
)

# A noul with described yes and no; at least one carries a description.
hazard = Question(
    "noul",
    Text("Does the report describe a danger to people?"),
    [{"name": "yes", "description": "Someone could be hurt"}, "no"],
)

# A structured wording: each string value is a template from the one slot list.
kind = Question(
    "choice",
    Structured(
        {"task": "What kind of report is {body}?", "examples": ["a lamp out", "a pothole"]},
        [{"name": "body", "origin": {"ref": "t:state.body"}}],
    ),
    ["lighting", "road", "other"],
)
```

## Handing off to the next model

A design's context block says what a turn hands the next model in a conversation, and `handoff(record)` reads it back; `transcript(record)` gives every call of the turn and its conclusion. `ommat.conversation`, outside the compatibility promise, renders both and words nothing:

```python
from ommat import Context, ContextEntry, Text, handoff, transcript
from ommat.conversation import handoff_text, to_chat_completions

# Each entry flags t:state or a d: ref and words it; one slot names the flagged ref.
context = Context(
    [
        ContextEntry(
            "d:refund",
            Text("Refund requested: {r}", [{"name": "r", "origin": {"ref": "d:refund"}}]),
        )
    ],
    plan=True,  # the plan's kind, verdict, reply and reason
    steps=True,  # the steps, only under a passed verdict
    record=True,  # the design and the record hash
)
# Design(..., context=context); after decide:
#   instructions = handoff_text(handoff(decision.record))
#   messages = to_chat_completions(transcript(decision.record))  # after your user message
# With ommat[pydantic-ai]:
#   from pydantic_ai.capabilities import ProcessHistory
#   from ommat.conversation.pydantic_ai import drop_earlier_bookkeeping, to_model_messages
#   history += to_model_messages(transcript(decision.record), model_name="ommat")
#   agent.run(None, message_history=history)  # runs the pending calls with the agent's tools
#   Agent(..., capabilities=[ProcessHistory(drop_earlier_bookkeeping)])  # earlier turns' calls out
```

See the how-to [Hand off what a turn settled](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/how-to/hand-off-context.md).

## Rules of the engine

- An unknown is a marker, `Unsure`, `Unavailable` or `Unasked` (all `Undetermined`), never `None` or a quiet no. Test reads with `isinstance(value, Undetermined)` before using them, in rules and in the policy; `if value:` on a marker raises. A rule passes a read's marker on with its own cause by returning it; `r.require(ref)` stops instead, with cause `required` naming the read.
- A rule reads a `q:` answer as an `Answer`, its `labels` and `probabilities`, not the mapping a binding returns. A rule that raises makes its ref `Unavailable` with `raised` and the turn goes on; the traceback is in the record's `unhashed["exceptions"]`, keyed by the entry's log index. In a test, assert that it is empty.
- Choose a label with `ommat.helpers.pick_with_margin(r, ref, bar, margin)` or `pick_with_thresholds`, not your own argmax: a tie at the top gives `Unsure` with `abstained`, even at `bar` and `margin` 0, where an argmax taking the first label in declared order picks one.
- `decide` calls no tool. The caller runs the steps of a plan whose verdict passed and whose kind is `execute`, on its own engine or with `ommat.helpers.run_plan`. Dispatch on `plan.kind`, never on the verdict alone: the guard passes a `broken` plan, which has no steps. See the [spec's Decide section](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#decide).
- Define the design's rules, `when`s and policy in an importable module, not in `__main__`, so a record replays under it in another process with no `design-mismatch`. The [spec's Replay section](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#replay) says how.

## Testing

- Bind a stand-in: an `async def binding(batch, *, seconds) -> CallResult` giving a fixed answer for each `q:` ref in each `SourceCall`'s `questions`, probabilities keyed by label name, an extraction's answer through `ommat.bindings.extraction_value`. The README demo has one. A binding without the `seconds` keyword fails every call.
- `ommat.bindings.conformance.check(binding)`, outside the compatibility promise, holds a binding of your own to a provider to the protocol: it returns the failures, `[]` on a pass, and must be called outside a running event loop. Its `provenance` check asks a binding that sent a request to give each provenance detail a value or a reason in `missing`; build the result with `ommat.bindings.call_result` over one `RequestProvenance` per request.
- Assert on `decision.plan.kind`, `decision.verdict.kind` and the replay findings. To read an estimate a turn wrote, an `escalate` plan's working among them, take its entry from the record: `{i["ref"]: i for i in decision.record.log if i["item"] == "entry"}`; each ref is written once.
- Two design lints run the design with no model: `ommat.helpers.dead_questions(loaded)` reports, per question, whether its answer changed the plan or the verdict at any point it tried (`live`, `no-change-found` or `not-on-grid`), and `ommat.helpers.unsure_to_execute(loaded)` reports each point where the policy, given a `d:` read set to `Unsure`, returns an `execute` plan. Each takes `values`, a mapping from ref to a list of values, for a read its types and grid cannot fill: `dead_questions` fills `t:state` and what sources provide, `unsure_to_execute` the policy's declared reads, `d:` included.

## Pitfalls

- **load refuses what 0.4.0 leaves out:** decision tables, data rules, plan declarations, candidate sets of records and per-candidate questions. A `choice` over a `list<string>` candidate set loads, `{"fixed": [...], "candidates": ref, "at": k}` with `at` placing the candidates after the first `k` fixed labels; its question is left out with a marker of its own when the set is empty or a marker, or its labels overflow or collide. See the [spec's intro](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md) and [the candidate-valued label set](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#the-candidate-valued-label-set).
- **A `noul` declares no labels, or `yes` then `no` with a description.** A plain `["yes", "no"]` is refused at load: an undescribed `noul` leaves `labels` unset.
- **Nothing carries between turns.** Each turn stands alone; the caller passes what a later turn needs, such as an earlier step's result, in its input state, in fields declared in `state`: decide drops an undeclared field without a word. See [Continuing a workflow](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#continuing-a-workflow).
- **Declare every read.** A rule that reads a ref outside its `reads` makes its `d:` ref `Unavailable` (`undeclared-read`); the policy's makes the plan `broken`. Catching it does not help.
- **A step with no requirement of its own refuses the plan**, unless the design sets `default_deny=False`. Give each step the `Requirement`s it rests on.
- **Rules, `when`s, the policy and the codec are sync.** An `async def` one loads, but at run its ref is `Unavailable` with `invalid-return`, or the plan `broken` for the policy. An `async def` observer is refused at entry with `EntryRefusal`.
- **A handoff's text is the input's own.** An entry whose slot reads `t:state` or a source's `t:` ref carries what the user or a tool wrote; so does a step's board-ref argument, which `handoff_text` and the transcript show with its value. Fence `handoff_text` as data in a user message, rather than placing it as instructions, unless every slot and argument reads values your code settled. Load refuses a slot over an `identifier` field, not an argument.
- **A transcript's pending calls name your tools.** A step's tool is given as the design declares it, so declare the names your agent's tools have; a name a provider refuses, or `decide`, is a `ValueError`. A plan with a step that binds an earlier result has no pending calls: run it with `run_plan`. Leave ommat's bookkeeping, the source calls and decide, out of the history a later turn reads as `t:state`: `is_bookkeeping(message)` tells it, and `Agent(capabilities=[ProcessHistory(drop_earlier_bookkeeping)])` drops an earlier turn's. The pending calls are not bookkeeping: the agent ran them.
- **A slot's origin is a ref**, with a field path through records, `t:state.sender.name`, but no list index; give each value its own ref.
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
- A described `noul` or a `Structured` wording goes only to a provider whose capabilities carry it (`carries_noul_descriptions`, `carries_structured_instructions`); otherwise that question alone is `unsupported` and the source's other questions are sent.
- A `choice` answer keeps its probabilities only: idia's chosen label is dropped, so where a provider broke a tie the choice helpers abstain.
- `oracle_binding` follows idia while idia is 0.x, so it may change in any minor release; it is outside ommat's compatibility promise.
- For idia itself (providers, credentials, errors), install idia's own skill from [skills/idia](https://github.com/CRSiegfried/idia/tree/v0.1.0/skills/idia).
- Walk through a real binding with the tutorial [Decide with real models through idia](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/tutorials/decide-with-idia.md), and write one of your own with the how-to [Add a source](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/how-to/add-a-source.md).

## Details

- [The spec](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md): what ommat does, and what 0.4.0 leaves out.
- [Write your first design](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/tutorials/write-a-design.md) and [Decide with real models through idia](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/tutorials/decide-with-idia.md).
- [The README demo](https://github.com/CRSiegfried/ommat/blob/v0.4.0/README.md#a-first-decision).
- [The public API](https://github.com/CRSiegfried/ommat/blob/v0.4.0/CONTRIBUTING.md#ground-rules-for-changes): the names under the compatibility promise.
