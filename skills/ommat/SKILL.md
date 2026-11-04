---
name: ommat
description: Use when turning LLM or decision-model answers into a replayable decision with ommat, the Python decision engine; when writing or changing a design (sources, questions, rules, the policy); when calling load, decide or replay; when routing an undecided turn with escalation; when making agent decisions auditable or replayable with ommat; or when binding ommat to models through ommat[idia].
---

# ommat

ommat turns many small model answers into one decision. A model answers small questions; your rules, plain Python functions or tables, turn the answers into estimates; the policy, one of the rules, makes its estimate the decision; and a record replays the turn with no model call. ommat stops at the decision: it names no tool, step or action, and the caller acts.

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
- `rules`: each keyed by its name and providing `d:<name>`, a function from a read view to one estimate, with its `reads` declared, or a `Table` of rows over its inputs. Source ids, rule ids and parameter paths share one namespace, so a rule and a source cannot share a name.
- `policy`: the `d:` ref of one declared rule, `policy="d:route"`. Its estimate is the turn's decision. Its ref has no field path and is not grounded; a rule may read it as it reads any `d:` ref.
- bindings: async functions the caller passes to decide, keyed by oracle name (`default` for a source that names none).

```python
from ommat import decide_sync, load, replay_sync

loaded = load(DESIGN)  # raises InvalidDesign with every failure
decision = decide_sync(loaded, state, {"default": binding})
decision.estimate, decision.record, decision.design_hash
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

From format 7, targets name, for a reader, the property a question asks about or a rule estimates, and the ones a question bears on; they gate nothing and leave the question's identity alone:

```python
from ommat import Question, Target, Text

targets = {
    "tg:danger": Target("Whether people are at risk where the report was made", values="yes or no"),
    "tg:priority": Target("How soon a crew should go out"),
}
danger = Question(
    "noul",
    Text("Could someone be hurt where this report was made?"),
    target="tg:danger",  # each id is a key of the design's targets
    latent_targets=[{"target": "tg:priority", "role": "formative"}],  # or reflective, unknown
)
# Design(..., targets=targets); a rule names what it estimates: Rule(..., target="tg:priority")
```

## Handing off to the next model

`transcript(record)` gives every call of a turn and its decision, unworded. `ommat.conversation`, outside the compatibility promise, renders it and words nothing; you word any instructions yourself, from `decision.estimate` and `escalation(record)`:

```python
from ommat import transcript
from ommat.conversation import to_chat_completions

# After decide:
#   messages = to_chat_completions(transcript(decision.record))  # after your user message
# With ommat[pydantic-ai]:
#   from pydantic_ai.capabilities import ProcessHistory
#   from ommat.conversation.pydantic_ai import drop_earlier_bookkeeping, to_model_messages
#   history += to_model_messages(transcript(decision.record), model_name="ommat")
#   agent.run(user_message, message_history=history)
#   Agent(..., capabilities=[ProcessHistory(drop_earlier_bookkeeping)])  # earlier turns' calls out
```

See the how-to `docs/how-to/hand-off-a-turn.md` in the ommat repository.

## Rules of the engine

- An unknown is a marker, `Unsure`, `Unavailable` or `Unasked` (all `Undetermined`), never `None` or a quiet no. Test reads with `isinstance(value, Undetermined)` before using them, in rules and in the policy; `if value:` on a marker raises. A rule passes a read's marker on with its own cause by returning it; `r.require(ref)` stops instead, with cause `required` naming the read.
- A rule reads a `q:` answer as an `Answer`, its `labels` and `probabilities`, not the mapping a binding returns. A rule that raises makes its ref `Unavailable` with `raised` and the turn goes on; the traceback is in the record's `unhashed["exceptions"]`, keyed by the entry's log index. In a test, assert that it is empty.
- Choose a label with `ommat.helpers.pick_with_margin(r, ref, bar, margin)` or `pick_with_thresholds`, not your own argmax: a tie at the top gives `Unsure` with `abstained`, even at `bar` and `margin` 0, where an argmax taking the first label in declared order picks one.
- `decide` calls no tool and names no step or action. It returns a `Decision`; its `estimate` is the policy's. A value is the decision: the caller acts on it. A marker means undecided: route it with `escalation(decision.record)`, which gives the marker's `kind` and `cause`, the `root_cause` (following `required` causes to the entry they name), `defect` and the `record_hash`. Route by `escalation(...)["defect"]`: true is a defect to fix (a root cause of `raised`, `invalid-return` or `undeclared-read` naming a rule or the codec); false is for a person or a later turn to settle. See the [spec's Decide section](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#decide).
- Define the design's rules, `when`s and policy in an importable module, not in `__main__`, so a record replays under it in another process with no `design-mismatch`. The [spec's Replay section](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#replay) says how.

## Testing

- Bind a stand-in: an `async def binding(batch, *, seconds) -> CallResult` giving a fixed answer for each `q:` ref in each `SourceCall`'s `questions`, probabilities keyed by label name, an extraction's answer through `ommat.bindings.extraction_value`. The README demo has one. A binding without the `seconds` keyword fails every call.
- `ommat.bindings.conformance.check(binding)`, outside the compatibility promise, holds a binding of your own to a provider to the protocol: it returns the failures, `[]` on a pass, and must be called outside a running event loop. Its `provenance` check asks a binding that sent a request to give each provenance detail a value or a reason in `missing`; build the result with `ommat.bindings.call_result` over one `RequestProvenance` per request.
- Assert on `decision.estimate` and the replay findings. To read an estimate a turn wrote, take its entry from the record: `{i["ref"]: i for i in decision.record.log if i["item"] == "entry"}`; each ref is written once.
- One design lint runs the design with no model: `ommat.helpers.dead_questions(loaded)` reports, per question, whether its answer changed the decision at any point it tried (`live`, `no-change-found`, `not-on-grid`, or `not-grounded` for a per-candidate question no point grounded). It takes `values`, a mapping from ref to a list of values, for a read its types and grid cannot fill: it fills `t:state` and what sources provide. A `values` list for a label question with no grid entry, other than a candidate-valued one, narrows it to those labels, and the report's `narrowed` names it. It raises `ValueError` for a point whose policy entry carries an entry cause. On `main`, when `dead_questions` has too many points to run, `ommat.evaluation.question_reach_sync(records, loaded, values=None)` asks the same question from recorded turns: it sets each asked answer alone to each of its points, replays, and reports each question `live` with a witness, `not-determined`, `no-change-found` or `never-asked`. Its `values` is `dead_questions`' restricted to `q:` refs: an extraction question with no grid entry needs a list of values there, else a `ValueError` names it, and a label question with no grid entry may be narrowed.

- To look at a turn, write each `decision.record.document` to a `.json` file, with `loaded.document` beside them, and run `python -m ommat.lens <file or folder> --open`: each round's calls and entries, the answers as percentages, and each value's lineage. The page holds the records whole, user text included, so keep it as you keep them. `ommat.lens` is outside the compatibility promise.

## Pitfalls

- **load refuses what 0.4.0 leaves out:** candidate sets of records and grounded rules, tables and `t:` sources, and every decision table, all of which `main` runs. There a decision table loads: write its cells in the notation (`-`, `Unsure`, `>= urgent/bar`, `{a, b}`, a label) or tagged, and a label the notation would misread, such as `-`, as `{"is": "-"}`. A `choice` over a `list<string>` candidate set loads, `{"fixed": [...], "candidates": ref, "at": k}` with `at` placing the candidates after the first `k` fixed labels; its question is left out with a marker of its own when the set is empty or a marker, or its labels overflow or collide. On `main`, a table cell over a pick of that question may name a fixed label, such as `none`, but never a candidate. See the [spec's intro](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md) and [the candidate-valued label set](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#the-candidate-valued-label-set).
- **A per-candidate question is read as a `dict`.** Declare the source `grounded="d:x"` over a `list<string>` candidate set and put a slot `{"name": "c", "origin": {"candidate": "d:x"}}` in the question's wording: it is asked once per candidate under `q:y["<candidate>"]`, beside the source's other questions, asked once. A rule reads `q:y` and gets `{candidate: Answer or marker}` in the set's order, or the set's need marker; an empty set asks nothing and reads `{}`. Pick each candidate's label with `ommat.helpers.pick_each_with_margin(r, "q:y", bar, margin)` or `pick_each_with_thresholds`: a new `dict` from each candidate to its label, to `Unsure` with `abstained` naming that candidate's grounded ref, or to its marker; return your own value from it, or pass one candidate's marker on. Load refuses an extraction asked per candidate, one over a candidate-valued label set, and a need or slot ref on `q:y`. See [Per-candidate questions](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#per-candidate-questions).
- **On `main`, ground the rest of the design over the set too.** A set may be a list of records: `candidates="name"` names its candidate field, a `string`, which every label, grounded ref and `in` cell uses. A rule or table declared `grounded="d:x"` runs once per candidate, reading `candidate`, `candidate.f` and the candidate's own entry of a ref grounded over the same set, and writes `d:r["<candidate>"]`; a reader of `d:r` gets the `dict`. A source grounded over the set may provide `t:` refs, one per candidate: its binding gets each grounded ref in `refs` and reads the candidate with `ommat.bindings.candidate_of(ref)`. A per-candidate question may slot a ref grounded over its own set. Load refuses a `candidate` read outside a grounded component and a grounded declaration that is itself a candidate set.
- **A `noul` declares no labels, or `yes` then `no` with a description.** A plain `["yes", "no"]` is refused at load: an undescribed `noul` leaves `labels` unset.
- **Nothing carries between turns.** Each turn stands alone; the caller passes what a later turn needs, such as what the caller did on an earlier decision, in its input state, in fields declared in `state`: decide drops an undeclared field without a word. See [Continuing a workflow](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#continuing-a-workflow).
- **Declare every read.** A rule, the policy included, that reads a ref outside its `reads` makes its `d:` ref `Unavailable` (`undeclared-read`); the turn is undecided when it is the policy's. Catching it does not help.
- **The policy is a rule.** Declare it in `rules`, point `policy` at its ref, and let it pass a read's marker on when it cannot decide. A design with no policy is refused at load.
- **Rules, `when`s and the codec are sync.** An `async def` one loads, but at run its ref is `Unavailable` with `invalid-return`; the turn is undecided when it is the policy's. An `async def` observer is refused at entry with `EntryRefusal`.
- **A transcript is the input's own.** It gives every answer and estimate as recorded, `identifier` fields included. Fence any text you build from it as data in a user message, rather than placing it as instructions, unless every field it reads is one your code settled.
- **A transcript is bookkeeping.** It renders the rounds' source calls and decide's call and its decision as tool calls and results; it holds no call for your tools. Leave it out of the history a later turn reads as `t:state`: `is_bookkeeping(message)` tells it, and `Agent(capabilities=[ProcessHistory(drop_earlier_bookkeeping)])` drops an earlier turn's. When a turn can repeat exactly, as a retry does, pass each rendering its own `turn=`, a `str` or an `int`, so no tool-call id repeats; a provider may refuse a repeated one.
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
- A label source's needs go as one JSON part keyed by their refs, in code-point order, each object inside in the order the binding receives it: a record value's fields in its type's declared order. Key order is part of what the model reads. To send the state shape a provider was measured on, declare it as one record need with its fields in that order and build the binding with `oracle_binding(dispatcher, state="t:state")`, which sends that need's value as the whole part; a label source needing it beside another need, images aside, is `unsupported`.
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
