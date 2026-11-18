---
name: ommat
description: Use when turning LLM or decision-model answers into a replayable decision with ommat, the Python decision engine; when writing or changing a design (sources, questions, rules, the policy); when calling load, decide or replay; when routing an undecided turn with escalation; when making agent decisions auditable or replayable with ommat; or when binding ommat to models through ommat[idia].
---

# ommat

ommat turns many small model answers into one decision. A model answers small questions; your rules, plain Python functions or tables, turn the answers into estimates; the policy, one of the rules, makes its estimate the decision; and a record replays the turn with no model call. A source's binding may call any tool, one that writes included, and later rules read what it returned.

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
- `parameters`: each tunable number, such as a bar, is a leaf `Parameter(type, value)` keyed by its path, read with `r.param(path)` and declared in the rule's `params`. Never capture one in a closure or fold it into a rule's `version`: the lints and `switching_values` see only the tree. A design per bars value is a new design hash; if loading per turn matters, cache the `LoadedDesign` per bars value. A builder argument may be a `t.label(...)` parameter, read the same way, only when every one of its values is valid in every design it builds, since `switching_values` tries each value; one that also changes the design's sources or a rule's `reads` stays a builder argument, with a module-level function per value. A `r.param` path missing from the rule's `params` makes its entry `Unavailable` with `undeclared-read` naming the rule and then the path, which has no `q:`, `t:` or `d:` prefix.
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

Each `decide_sync` or `replay_sync` call builds an event loop of its own. In a loop over many turns, open one `asyncio.Runner` and pass `runner=runner` to each call, or await `decide_many`; thousands of loops use up Windows's ephemeral ports.

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
    "tg:danger": Target("The risk to people where the report was made", values="yes or no"),
    "tg:priority": Target("The urgency of sending a crew out"),
}
danger = Question(
    "noul",
    Text("Could someone be hurt where this report was made?"),
    target="tg:danger",  # each id is a key of the design's targets
    latent_targets=[{"target": "tg:priority", "role": "formative"}],  # or reflective, unknown
)
# Design(..., targets=targets); a rule names what it estimates: Rule(..., target="tg:priority")
```

To vary a design, `dataclasses.replace` its parts (`Design`, `Source`, `Rule`, `Question`) and `load` the result; load names anything the change left dangling, such as a read of a removed rule. Don't `copy.deepcopy` a design: it fails.

## Authoring a design as TOML

`ommat.authoring.design_from(tomllib.load(f))` reads a TOML file of a design's declarations and decision tables as the `Design` you would write in Python, with the same hash; rule logic stays in Python. TOML is advised for a design's declarations and tables, Python is always accepted, and one design or slice is wholly one form. A file is trusted as source code, because the reader imports each rule's `fn`. Read [toml.md](toml.md) before you write or review one.

## Handing off to the next model

When a turn hands on to a generative model, as when the design has done what it can and an agent finishes the request, render the record with `ommat.conversation`, and don't rebuild its history from the record's `t:` values. It gives the turn as an agent's workflow, minimal: each source call, your tools' reads and writes included, as a tool call with empty arguments and its answers (a question's top label and probability, a `t:` value whole, a marker as its kind and cause code), then `decide`, whose arguments are the board entries the policy reads when you pass the loaded design, and whose result is the decision or why the turn is undecided. `ommat.conversation`, outside the compatibility promise, words nothing; you word any instructions yourself, from `decision.estimate` and `escalation(record)`. `transcript(record)` is deprecated:

```python
from ommat.conversation import to_chat_completions

# After decide:
#   messages = to_chat_completions(decision.record, loaded)  # after your user message
# With ommat[pydantic-ai]:
#   from pydantic_ai.capabilities import ProcessHistory
#   from ommat.conversation.pydantic_ai import drop_earlier_bookkeeping, to_model_messages
#   history += to_model_messages(decision.record, loaded, model_name="ommat")
#   agent.run(user_message, message_history=history)
#   Agent(..., capabilities=[ProcessHistory(drop_earlier_bookkeeping)])  # earlier turns' calls out
```

See the how-to `docs/how-to/hand-off-a-turn.md` in the ommat repository.

The history carries each `t:` value whole, so a binding's values set most of its size. Return a tool's result as a structured value, never a JSON string, which is escaped again at each layer; return each fact once, not the raw payload beside the parsed list; and drop the transport around it, but keep every field a rule reads or a what-if may re-read, since replay answers from the record. See "Shape a tool's value" in the how-to `docs/how-to/add-a-source.md`.

## Rules of the engine

- An unknown is a marker, `Unsure`, `Unavailable` or `Unasked` (all `Undetermined`), never `None` or a quiet no. Test reads with `isinstance(value, Undetermined)` before using them, in rules and in the policy; `if value:` on a marker raises. A rule passes a read's marker on with its own cause by returning it; `ommat.helpers.determined(r, *refs)` gives several reads' values as a tuple, or the first marker by precedence unchanged, so one test guards them all. `r.require(ref)` stops instead, with cause `required` naming the read.
- A rule reads a `q:` answer as an `Answer`, its `labels` and `probabilities`, not the mapping a binding returns. A rule that raises makes its ref `Unavailable` with `raised` and the turn goes on; the traceback is in the record's `unhashed["exceptions"]`, keyed by the entry's log index. The turn also emits an `ommat.DefectWarning`; in a test suite, make it an error (see the `decide` bullet).
- Choose a label with `ommat.helpers.pick_with_margin(r, ref, bar, margin)` or `pick_with_thresholds`, not your own argmax: a tie at the top gives `Unsure` with `abstained`, even at `bar` and `margin` 0, where an argmax taking the first label in declared order picks one. For the top label with no bar, and its probability, use `ommat.helpers.top(r, ref)`; pass `tie="first"` to take the first tied label in `labels` instead of abstaining. For a `noul`'s p(yes), use `ommat.helpers.mass(r, ref, ["yes"])`.
- `decide` returns a `Decision`; its `estimate` is the policy's. A value is the decision: the caller acts on it. A marker means undecided: route it with `escalation(decision.record)`, which gives the marker's `kind` and `cause`, the `root_cause` (following `required` causes to the entry they name), `defect` and the `record_hash`. Route by `escalation(...)["defect"]`: true is a defect to fix (a root cause of `raised`, `invalid-return` or `undeclared-read` naming a rule or the codec); false is for a person or a later turn to settle. `ommat.defects(decision.record)` lists every entry with such a cause, whatever the policy gave, so a policy that always returns a value does not hide a rule that raised; it is `None` for a record before format 8. Such a turn also emits one `ommat.DefectWarning` once its record closes, carrying `record` and `defects` and naming each defect with its traceback; set `filterwarnings = ["error::ommat.DefectWarning"]` in pytest config, or `warnings.simplefilter("error", ommat.DefectWarning)`, to make every such turn raise with its record, rather than asserting `unhashed["exceptions"]` is empty. `record.entries` maps each ref to its entry item. See the [spec's Decide section](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#decide).
- Lay out a design module so it reads top to bottom: a docstring map of its sources and its rules in graph order, each rule function named for the ref it provides, refs as literals, a table's inputs named and its rows built by name, one bar per decision, and a read-back into your own type. See the how-to `docs/how-to/lay-out-a-design.md` in the ommat repository.
- Define the design's rules, `when`s and policy as functions at module level in an importable module, not as closures or in `__main__`, so a record replays, and a stored design reloads, by import in another process with no `design-mismatch`. The [spec's Replay section](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#replay) says how.

## Testing

- `ommat.bindings` is the standard library alone and imports no provider; each provider binding is a submodule behind its extra, such as `ommat.bindings.idia`. A rule that keeps providers inside one connector bans those submodules, not `ommat.bindings`.
- Bind a stand-in, or a fact the caller holds, with `ommat.bindings.fixed({ref: value})`: it answers each asked ref it holds, a `q:` ref as probabilities keyed by label name, an extraction's answer through `ommat.bindings.extraction_value`, a `t:` ref as a value of its type, and gives an asked ref it does not hold `unsupported`. Bind each oracle its own, as in `{"default": fixed({"q:refund": {"yes": 0.9, "no": 0.1}}), "clock": fixed({"t:today": today.isoformat()})}`. A binding of your own, for what `fixed` cannot do such as failing the call, is `async def binding(batch, *, seconds) -> CallResult`, answering only the refs and questions each `SourceCall` asks, since an unasked key fails the call; without the `seconds` keyword it fails every call.
- `ommat.bindings.conformance.check(binding)`, outside the compatibility promise, holds a binding of your own to a provider to the protocol: it returns the failures, `[]` on a pass, and must be called outside a running event loop. Its `provenance` check asks a binding that sent a request to give each provenance detail a value or a reason in `missing`; build the result with `ommat.bindings.call_result` over one `RequestProvenance` per request.
- Assert on `decision.estimate` and the replay findings. To read an estimate a turn wrote, take its entry from the record, `decision.record.entries["d:x"]`, which holds a `value` or a `marker`; read a record through `entries`, `escalation`, `defects` and `ommat.evaluation.answers` rather than scanning `log`. See the how-to `docs/how-to/read-a-record.md` in the ommat repository.
- One design lint runs the design with no model: `ommat.helpers.dead_questions(loaded)` reports, per question, whether its answer changed the decision at any point it tried (`live`, `no-change-found`, `not-on-grid`, or `not-grounded` for a per-candidate question no point grounded). It takes `values`, a mapping from ref to a list of values, for a read its types and grid cannot fill: it fills `t:state` and what sources provide. A `values` list for a label question with no grid entry, other than a candidate-valued one, narrows it to those labels, and the report's `narrowed` names it. It raises `ValueError` for a point whose policy entry carries an entry cause. When `dead_questions` has too many points to run, `ommat.evaluation.question_reach_sync(records, loaded, values=None)` asks the same question from recorded turns: it sets each asked answer alone to each of its points, replays, and reports each question `live` with a witness, `not-determined`, `no-change-found` or `never-asked`. Its `values` is `dead_questions`' restricted to `q:` refs: an extraction question with no grid entry needs a list of values there, else a `ValueError` names it, and a label question with no grid entry may be narrowed.
- To look at a turn, write each `decision.record.document` to a `.json` file, with `loaded.document` beside them, and run `python -m ommat.lens <file or folder> --open`: what the turn did in plain words (the decision, each table's rows, each source asked or not and why), an overview of the design, a slice of its graph around any node with the selected turn's entries marked on it, its targets and its tables' rows, and the whole graph by round with each turn's entries marked against their bounds, the lineage from the decision to its root, and a table of turns. The page holds the records whole, user text included, so keep it as you keep them. `ommat.lens` is outside the compatibility promise. For text you can grep or diff, one node, call or entry per line, run `python -m ommat.outline <file or folder>`, or add `--json` for the outline dicts; `ommat.outline` is outside the promise too.

## Pitfalls

- **Write a decision table's cells in the notation.** A decision table loads: write its cells in the notation (`-`, `Unsure`, `>= urgent/bar`, `{a, b}`, a label) or tagged, and a label the notation would misread, such as `-`, as `{"is": "-"}`. A `-` cell matches an `Unasked` input too, so a table meant to stay `Unasked` when a question was skipped holds `Known` over that input in its last row; see [the spec's decision tables](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#decision-tables). A `choice` over a `list<string>` candidate set loads, `{"fixed": [...], "candidates": ref, "at": k}` with `at` placing the candidates after the first `k` fixed labels; its question is left out with a marker of its own when the set is empty or a marker, or its labels overflow or collide. A table cell over a pick of that question may name a fixed label, such as `none`, but never a candidate. See the [spec's intro](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md) and [the candidate-valued label set](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#the-candidate-valued-label-set).
- **A per-candidate question is read as a `dict`.** Declare the source `grounded="d:x"` over a `list<string>` candidate set and put a slot `{"name": "c", "origin": {"candidate": "d:x"}}` in the question's wording: it is asked once per candidate under `q:y["<candidate>"]`, beside the source's other questions, asked once. A rule reads `q:y` and gets `{candidate: Answer or marker}` in the set's order, or the set's need marker; an empty set asks nothing and reads `{}`. Pick each candidate's label with `ommat.helpers.pick_each_with_margin(r, "q:y", bar, margin)` or `pick_each_with_thresholds`: a new `dict` from each candidate to its label, to `Unsure` with `abstained` naming that candidate's grounded ref, or to its marker; return your own value from it, or pass one candidate's marker on. Load refuses an extraction asked per candidate, one over a candidate-valued label set, and a need or slot ref on `q:y`. See [Per-candidate questions](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#per-candidate-questions).
- **Ground the rest of the design over the set too.** A set may be a list of records: `candidates="name"` names its candidate field, a `string`, which every label, grounded ref and `in` cell uses. A rule or table declared `grounded="d:x"` runs once per candidate, reading `candidate`, `candidate.f` and the candidate's own entry of a ref grounded over the same set, and writes `d:r["<candidate>"]`; a reader of `d:r` gets the `dict`. A source grounded over the set may provide `t:` refs, one per candidate: its binding gets each grounded ref in `refs` and reads the candidate with `ommat.bindings.candidate_of(ref)`. A per-candidate question may slot a ref grounded over its own set. Load refuses a `candidate` read outside a grounded component and a grounded declaration that is itself a candidate set.
- **A `noul` declares no labels, or `yes` then `no` with a description.** A plain `["yes", "no"]` is refused at load: an undescribed `noul` leaves `labels` unset.
- **Nothing carries between turns.** Each turn stands alone; the caller passes what a later turn needs, such as what the caller did on an earlier decision, in its input state, in fields declared in `state`: decide drops an undeclared field without a word. See [Continuing a workflow](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#continuing-a-workflow).
- **Set `round_bound` generously.** It stops a runaway turn; it is not the turn's expected depth. A design that acts through a chain of sources takes a round per link, so set it well above the longest chain, as `RunSettings(round_bound=32)`; at the bound, pending sources are `Unasked` with `round-bound`.
- **Give each source the dependencies it really has, and no others.** Interpret asks a source in the first round after its needs, its requirement's ref and, for a grounded source, its candidate set are on the board, so a source that depends on nothing but the state is asked in round 1; an extra need also goes to the binding. A call that should happen only when decided, every write and any read that is costly, sensitive or has a side effect, takes a source requirement on the one estimate that says to make it, other conditions such as a part's port folded into it, and needs only its arguments: a need does not hold a source back once it holds a value, `null` included. Never gate a question on another part's decision. A round's sources with no oracle are one call, and the idia binding sends those sharing a state as one request.
- **Declare every read.** A rule, the policy included, that reads a ref outside its `reads` makes its `d:` ref `Unavailable` (`undeclared-read`); the turn is undecided when it is the policy's. Catching it does not help.
- **Combine small designs with `ommat.helpers.combine`.** Write each workflow as a `Part(design, ports)`, its ports the `d:` refs it reads from another part, named under its own prefix, and roll them up with `combine(*parts, bind={port: rule_or_ref}, policy=top_rule)`. Ask every part's questions at once and let the workflow's rules read the selector's estimates through its ports; gating its sources on the selector costs a round and a call. `combine` renames nothing, so give each part its own id prefix, its own parameter paths and one `state` type; a source, rule, type, group or target two parts declare alike under one id is kept once, and it raises `InvalidCombination` listing every clash. For a question several parts ask alike, give it a source of its own, with the same id, needs, oracle and question in every part and the rules behind its needs declared alike too; the combined design asks it once, and each part reads it under its own bar. Run a part alone with `standalone(part, {port: value})`, or with `given=[port]` to take the port's value each turn from the source `ports`, bound at decide. When a part was measured on another state shape, keep it: a rule over the state alone builds the part's view, such as `d:email/view` holding `{"request": current_turn}`, its questions need that view under an oracle of their own, and that oracle's binding names it, `oracle_binding(provider, state="d:email/view")`; the rule fires in round 0, so the questions are still asked in round 1, in a request of their own. See [the spec's combining designs](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#combining-designs).
- **The policy is a rule.** Declare it in `rules`, point `policy` at its ref, and let it pass a read's marker on when it cannot decide. A design with no policy is refused at load.
- **Rules, `when`s and the codec are sync.** An `async def` one loads, but at run its ref is `Unavailable` with `invalid-return`; the turn is undecided when it is the policy's. An `async def` observer is refused at entry with `EntryRefusal`.
- **A turn history is the input's own.** It gives each `t:` value as recorded, identifying fields included. Fence any text you build from it as data in a user message, rather than placing it as instructions, unless every field it reads is one your code settled.
- **To share a turn, redact it under your own policy.** `from ommat import redact` and call `redact(record, root_key, policy)`, `root_key` a secret 32-byte `bytes` you keep, the result JSON bytes; it takes the policy every time: a mapping from a ref, or `ref.field.path`, to `open`, `identifier` or `content`, or an object of fields. Everything it names nothing for is digested, numbers, labels and answers included, so name each value you mean to show, such as `{"d:act": "open", "t:state": {"ticket_id": "identifier"}}`. A null passes unchanged unless it lies within a value digested whole. A ref or field path the record lacks, or a class name other than `open`, `identifier` or `content`, raises `ValueError`. The view's `applied` is the policy in full, and passing it back as the policy repeats the view; its digests are keyed to it, so two policies' views of one turn never share a digest. See [the spec's redacted view](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md#the-redacted-view).
- **A turn history is bookkeeping.** It renders the rounds' source calls, a source's call to your tool included, and decide's call and its decision as tool calls and results; a call you make after decide is not in it. Leave it out of the history a later turn reads as `t:state`: `is_bookkeeping(message)` tells it, and `Agent(capabilities=[ProcessHistory(drop_earlier_bookkeeping)])` drops an earlier turn's. When a turn can repeat exactly, as a retry does, pass each rendering its own `turn=`, a `str` or an `int`, so no tool-call id repeats; a provider may refuse a repeated one.
- **A value of several shapes is one record type**, a `kind` label and nullable fields, `t.record(kind=t.label("exchange", "refund"), amount=t.nullable(t.number()))`, not a JSON string: its fields are typed and checked, and a rule or table reads one by its field path, `d:x.kind`.
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
- A binding receives a call's sources together, and `oracle_binding` sends those that share a state and options to the provider as one request, split only where the provider's limits require; a source whose joined request the provider's preflight refuses, or whose state differs, goes alone. The sources that name no oracle are one call each round, so they pack too. A source asking an extraction sends only its text and images, and its extractions must read one text and one reference date; ask label questions about other needs in a source of their own.
- A label source's needs go as one JSON part keyed by their refs, in code-point order, each object inside in the order the binding receives it: a record value's fields in its type's declared order. Key order is part of what the model reads. To send the state shape a provider was measured on, declare it as one record need with its fields in that order and build the binding with `oracle_binding(dispatcher, state="t:state")`, which sends that need's value as the whole part; a label source needing it beside another need, images aside, is `unsupported`.
- A described `noul` or a `Structured` wording goes only to a provider whose capabilities carry it (`carries_noul_descriptions`, `carries_structured_instructions`); otherwise that question alone is `unsupported` and the source's other questions are sent.
- A `choice` answer keeps its probabilities only: idia's chosen label is dropped, so where a provider broke a tie the choice helpers abstain.
- `oracle_binding` follows idia while idia is 0.x, so it may change in any minor release; it is outside ommat's compatibility promise.
- For idia itself (providers, credentials, errors), install idia's own skill from [skills/idia](https://github.com/CRSiegfried/idia/tree/v0.1.0/skills/idia).
- Walk through a real binding with the tutorial [Decide with real models through idia](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/tutorials/decide-with-idia.md), and write one of your own with the how-to [Add a source](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/how-to/add-a-source.md).

## Details

- [The spec](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/spec/README.md): what ommat does and leaves out.
- [Write your first design](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/tutorials/write-a-design.md) and [Decide with real models through idia](https://github.com/CRSiegfried/ommat/blob/v0.4.0/docs/tutorials/decide-with-idia.md).
- [The README demo](https://github.com/CRSiegfried/ommat/blob/v0.4.0/README.md#a-first-decision).
- [The public API](https://github.com/CRSiegfried/ommat/blob/v0.4.0/CONTRIBUTING.md#ground-rules-for-changes): the names under the compatibility promise.
