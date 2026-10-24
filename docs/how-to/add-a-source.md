# Add a source

This guide wires a new oracle into a design: a source that asks it, a binding that answers
for it, and the checks that hold the binding to ommat's protocol. It assumes you have done
[Write your first design](../tutorials/write-a-design.md). The
[binding protocol](../spec/README.md#the-binding-protocol) in the spec states the protocol;
the guide shows where each rule bites.

There are two ways to answer a source. If idia has a provider for your model, bind it
through idia, as [Path A](#path-a-bind-an-idia-provider) shows. Otherwise write a binding of
your own, as [Path B](#path-b-write-a-binding-over-a-local-model) shows over a model run in
your own process, such as a named-entity model on torch.

The example's files are in `docs/how-to/add_a_source/`: `street_spotter.py` is a small
stand-in for a local span model, so the example runs with no model to download,
`span_binding.py` is the binding over it, `reports_design.py` the design that asks it and
`reports_run.py` the driver; `idia_path.py` and `idia_run.py` are Path A's, and need
`ommat[idia]`. The design sits in a module of its own because a component's module is part of
the design's hash, and one defined in the running script, `__main__`, replays with
`design-mismatch`. Run the driver from that folder with `python reports_run.py`, in an
environment where ommat is installed.

## Declare the source

A source names the refs it provides, the board values it needs and the oracle that answers
it. The example's source asks two extraction questions of an oracle named `spans`:

```python
SOURCES = {
    "places": Source(
        {
            "q:places/street": Question(
                "extraction", Text("street"), None, value_type="text", text={"ref": "t:state.text"}
            ),
            "q:places/streets": Question(
                "extraction",
                Text("street"),
                None,
                value_type="text",
                is_list=True,
                text={"ref": "t:state.text"},
            ),
        },
        oracle="spans",
        timeout=5,
    ),
}
```

- **What it provides.** Each `q:` ref maps to a `Question`; each `t:` ref, a value the
  oracle fetches rather than judges, maps to its type. A source provides `q:` refs or `t:`
  refs, never both.
- **`needs`.** The board refs whose values the binding receives. An extraction's `text` and
  `reference_date` refs join them on their own, so this source declares none.
- **`oracle`.** The name you bind at decide. A source with none is answered by the binding
  named `default`.
- **`timeout`.** The seconds a call to this source has. A batch takes the largest timeout
  its sources declare, else the run settings' `timeout`, 60 seconds by default.

A question's kind is `noul`, `choice`, `score` or `extraction`. The first three ask for a
probability for each label: `noul` for `yes` and `no`, `choice` and `score` for the labels
the question declares. A label is its name, such as `"delivery"`, or exactly
`{"name": "delivery", "description": "Where an order is"}`; load refuses a mapping of `name`
alone. An extraction asks for a typed value read from a text, and declares
four members of its own:

- `value_type`: `text`, `number`, `date`, `time`, `datetime` or `duration`;
- `is_list`: whether the answer is a list of such values;
- `text`: an origin, `{"ref": r}`, naming the board ref of the string the answer's spans
  index;
- `reference_date`: such an origin, naming the date a date written without a year is read
  against, or `None`.

Both refs name values of type exactly `string`. The
[extraction kind](../spec/README.md#the-extraction-kind) gives the value grammars.

The wording is what the oracle receives, so word it for the model that answers. A span model
takes an entity label, so this design's wording is `Text("street")`, sent as it is.

## Bind the oracle by name

`decide` takes a mapping from oracle name to binding:

```python
        decision = decide_sync(loaded, {"text": report}, {"spans": binding})
```

Give one binding for each oracle a source names, and one under `default` when some source
names none. A name missing from the mapping, or a key other than `default` that the design
does not declare, is an `EntryRefusal` before the turn starts.

Each round, the ready sources that share an oracle go to its binding as one batch, in one
call; a source that names none is a batch of its own. The binding may split a batch into as
many provider requests as it likes. A round's calls run at once, so one binding bound under
several names, or as `default`, is called concurrently with itself.
[The binding protocol](../spec/README.md#the-binding-protocol) states the batching and the
timeout each batch takes.

## What a call receives

A binding is an async function, `binding(batch, *, seconds)`. `ommat.bindings.OracleBinding`
is its type. `ommat.Binding` is another thing: a plan step's argument bound to an earlier
step's result.

`batch` is a tuple of `SourceCall`s in source id order. Each holds:

- `id`, the source's id, and `payload`, the source's `payload` from the design;
- `needs`, each need's value keyed by the need as declared, never a marker, since a source is
  called only when its needs are determined;
- `refs`, the `t:` refs it provides, in code-point order;
- `questions`, each `q:` ref's `RenderedQuestion`, in the code-point order of their refs.

A `RenderedQuestion` holds `kind`, `wording`, `labels`, the label names, and `descriptions`,
aligned with them. An extraction question also holds `value_type`, `is_list`, `text`, the
`needs` key that holds its text, and `reference_date`, the rendered date or `None`.
`wording` is a string, or a JSON object or array for a
[structured wording](../spec/README.md#a-structured-wording). The labels of a
[candidate-valued](../spec/README.md#the-candidate-valued-label-set) `choice` may differ from
call to call, so keep no labels by ref.

A need that holds an answer is the `Answer` or `ExtractionAnswer` itself, and an image is
bytes; `ommat.bindings.json_value` writes a need as JSON, and `image_need` reads an image.
Treat `needs` as read-only, and change no durable state: no cache on disk, no write to a
store.

The order is part of what ran: a provider may answer a question differently beside other
questions. A binding that asks one question per request, or frames its own order, names that
method in `elicitation`.

`seconds` is the time the call has. [Seconds and timeouts](#seconds-and-timeouts) says what
to do with it.

## What a call returns

A binding returns a `CallResult`, or a `Failed` when the whole call failed.

Build `values` from `call.refs` and `call.questions` alone: each ref the batch provides maps
to its value, and no other key. A ref left out is written `Unavailable` with `not-returned`; a
key the batch does not provide, such as a candidate-valued question the engine left out of the
batch, fails the whole call with `unexpected-ref`.

- A `noul`, `choice` or `score` answer is a `dict` from each label name to its probability,
  a number in [0, 1], with exactly the question's label names as keys. Probabilities are
  written as returned, whatever their sum.
- An extraction answer is exactly `value`, `guess`, `p_null`, `confidence` and `span`.
  Build it with `ommat.bindings.extraction_value(value, guess, p_null, confidence, span)`,
  which the binding protocol describes.
- A `t:` ref's value is a value of its declared type.
- Any ref may instead be one of the markers below.

`Failed(Cause(code))` fails the call, with `raised`, `timed-out`, `invalid-return` or
`unexpected-ref`, and every ref of the batch is `Unavailable` with that code. A binding that
raises fails the call with `raised`. The binding protocol lists the other returns that fail
the whole call; `call_result` checks none of them. Mark refs one by one where you can, so
the other refs stand.

Catch `Exception`, never `BaseException`: the cancellation that ends a call at its timeout,
or that cancels the turn, must propagate. Return once, after any retries of your own.

## Markers a binding returns

A binding never returns `Unsure`: that marker is a rule's, saying it read its evidence and
did not settle. A binding marks a ref `Unavailable`, with one of six causes; an `Unsure`, an
`Unasked` or any other cause is written `invalid-return`.

| Cause | When |
|---|---|
| `unsupported` | The binding will not put the question to any provider: its ref, kind or shape is outside what the provider takes. |
| `refused` | The provider declined to answer. |
| `invalid-answer` | A `q:` ref's reply the binding cannot parse, or one interpret would refuse: a value outside the grammar, a span that does not read its value. |
| `invalid-return` | A `t:` value the provider returned that the binding cannot use, one its declared type does not allow. |
| `timed-out` | The request that carried the ref ran out of the call's `seconds`. |
| `raised` | The request that carried the ref failed any other way. |

Build each as `Unavailable(Cause("unsupported"))`; interpret names the ref in the cause.
Give the reason as a line of `warnings`, such as
`q:places/streets: unsupported: the model takes its label as a string, not a structured
wording`. A `timed-out` or `raised` ref under a call that reached a provider needs one.

`unsupported` is the binding's own refusal. Return it for each ref the binding will not
ask; never raise for one, and never leave one out of `values`. When the binding asks no
provider at all, its result carries null provenance.

A model that looked and found nothing is not a marker; its answer is a value
([The extraction kind](../spec/README.md#the-extraction-kind)). A scalar with nothing found
is `None`, on which `extract` abstains. A list is `[]` when the model read the text and found
no items, which `extract` returns, or `None` when it gave no reading. This binding returns
`[]`.

## Seconds and timeouts

`seconds` is a positive `float`, the smaller of the batch's timeout and the time left to the
turn's deadline, when the run settings set one. Turn it into a deadline once, as the call
starts, and add no margin of your own:

```python
        deadline = time.monotonic() + seconds  # once, with no margin of its own
```

Every request, retry and wait the call makes ends by that deadline. At the deadline, ommat's
own timeout ends the call: the call is `Failed` with `timed-out`, every ref of the batch
`Unavailable` with it, and a result that comes later is never read. A per-ref `timed-out`
stands only when the binding returns before the call's end, as when it finds before sending a
request that the request cannot run in time. A request that runs to the deadline ends the
whole call.

Size a source's `timeout` against the model's latency, measured on texts as long as yours. A
binding that serialises its model, as Path B's does, runs a call's requests one at a time, and
queues requests from concurrent calls behind them, so a call costs about the number of
questions times the time of one request, and a request's time can grow faster than the text's
length: for one small named-entity model on a CPU, 0.14 s at 536 tokens and 18 s at 7,800. A
call past its timeout is `Failed` with `timed-out`, but a thread started by `asyncio.to_thread`
runs on, and `decide_sync` waits for it before it returns.

## What the record keeps

A call's provenance is hashed on its call item: `provider`, `model_requested` and
`model_reported`, `revision`, `runtime`, the code that ran the model, `elicitation`, the
method, `precision`, how the provider rounds probabilities, `prompt`, what the model was sent,
and `claimed`, the details the source asserts and cannot verify. `usage`, `attempts`,
`request_id`, `calibrated`, `missing`, `clients`, `warnings` and `truncated` go under the
record's `unhashed`, outside every hash. A binding that sends a request gives each detail a
value, or a reason in `missing` why it has none;
[The provenance profile](../spec/README.md#the-provenance-profile) lists the details and the
reasons. The call item also holds each
question's identity, kind, labels as declared, value type and list flag, and the entries hold
each answer.

`decision.record.log` is the hashed log as a list of plain objects, one per item, in order;
`decision.record.document` is the stored form, canonical JSON bytes. Each `entry` item names
its `ref` and holds its `value`, or its `marker`. The driver reads the questions' values from
them, for each extraction question:

```python
def answers(decision: Decision) -> dict[str, object]:
    """Each extraction question's value, or its marker's cause, from the record's entry items."""
    return {
        item["ref"]: item["value"]["value"] if "value" in item else item["marker"]["cause"]["code"]
        for item in decision.record.log
        if item["item"] == "entry" and item["ref"].startswith("q:")
    }
```

Replay answers each source from the record and calls no binding, so a recorded turn reruns
with no model. Keep the design document, `loaded.document`, beside each record: when a
replay reports `design-mismatch`, [Replay](../spec/README.md#replay) says how to find the
member that differs.

## Path A: bind an idia provider

Install the binding with `ommat[idia]`, as the README's install section shows, then wrap an
idia `Dispatcher` over one leaf, as `idia_path.py` does:

```python
def idia_bindings(provider: idia.Dispatcher) -> dict[str, OracleBinding]:
    """The design's one oracle, answered by an idia provider."""
    return {"spans": oracle_binding(provider)}
```

`oracle_binding` sends each source as one idia request, writes the record's provenance and
marks what idia cannot ask `unsupported`; [The idia binding](../spec/README.md#the-idia-binding)
states what it does and which dispatchers it refuses. It is outside ommat's compatibility
promise: while idia is 0.x it may change in any minor release when idia changes, and the
changelog says so. [Decide with real models through idia](../tutorials/decide-with-idia.md)
runs it end to end over two wire models.

### Load a leaf that needs it

A leaf that runs a model in your process loads its weights before its first call: build it
with `idia.leaf(name)`, await its `load()`, then wrap it in a `Dispatcher`. idia's
[providers spec](https://github.com/CRSiegfried/idia/blob/main/docs/spec/providers.md) says
which leaves load. `idia_run.py` does it for any leaf:

```python
async def provider(leaf: Any) -> Dispatcher:
    """Load the leaf when it has a load step, then wrap it in a dispatcher."""
    load_step = getattr(leaf, "load", None)
    if load_step is not None:
        await load_step()
    return Dispatcher(leaf)
```

It loads on one event loop, `asyncio.run(provider(leaf))`, and each `decide_sync` then runs on
a loop of its own. idia's
[dispatch spec](https://github.com/CRSiegfried/idia/blob/main/docs/spec/dispatch.md) holds
that one `Dispatcher` serves successive `asyncio.run` calls, and its providers spec that a client a leaf creates follows the
running loop. A leaf given a client of your own refuses a call on any other loop: let the leaf
create its client, or load and decide on one loop with `decide`. Close the dispatcher when you
are done, as the driver does, to release the leaf.

### Put what an oracle cannot answer in a source of its own

idia refuses a whole request when one of its questions is a kind the provider does not take,
so every ref of that source comes back `unsupported`: one `choice` beside three extractions,
asked of an extraction model, loses all four. A described `noul` or a structured wording the
provider does not carry is `unsupported` alone. A leaf may also refuse at the call what its
capabilities do not say, such as a value type; then every ref of the request is `raised`,
with the leaf's message as a line. Give each oracle sources that ask
only what its model answers, and a design that asks label kinds and extractions of different
models one dispatcher per oracle.

### Split a long text before deciding

`oracle_binding` takes only a dispatcher whose overflow policy is `Fail`, so idia neither
trims nor splits a text. One longer than the provider's input does not fit, and the request is
refused before any call: every ref of the source is `unsupported`, with a line such as
`places: unsupported: the state alone is estimated at 346 tokens, over the limit of 256`. The
idia binding sends each source as one request, so the caller splits the text and decides each
part in a turn of its own.

### Run it offline

`idia_run.py` runs Path A with no model: idia's scripted leaf declares capabilities like a
text span leaf's, extractions of text, with an input window of 256 tokens so that the second
report overflows it, and answers through the same street finder. From
`docs/how-to/add_a_source`, run `python idia_run.py`:

```text
conformance: []
inform | Logged for Elm Row, Mill Lane; no one street stands out. | passed
  answers: {'q:places/street': 'Elm Row', 'q:places/streets': ['Elm Row', 'Mill Lane']}
  replay findings: [] | leaf calls: 0
escalate | the report names no street | passed
  answers: {'q:places/street': 'unsupported', 'q:places/streets': 'unsupported'}
  warning: places: unsupported: the state alone is estimated at 346 tokens, over the limit of 256
  replay findings: [] | leaf calls: 0
```

The second report is longer than that window, so idia refuses it before any call.

### What Path A records

Path A writes what idia's outcome reports under ommat's names: the provider, the models
requested and reported, idia's `snapshot` as the revision, the leaf's runtime versions, the
client libraries' versions, and idia's own reasons for each detail it leaves null and the
details it claims. A local leaf's model id is its own word, so idia claims it, and its
revision too when the leaf was given a model rather than loading one; a wire alias has no
revision, with `not-pinned` and a `no-snapshot` line in `warnings`.

## Path B: write a binding over a local model

`span_binding.py` binds a local span model: it takes a text and an entity label and returns
the spans it finds, each with its offsets and a score. It calls no API and answers the same
way each time.

### Load the model once

Loading a model takes seconds. Load it once, before `decide`, and hand it to the binding:

```python
    model = Counted(StreetSpotter())  # load once, before decide
    binding = SpanBinding(model)
```

The binding asks the model for one shape, `SpanModel`:

```python
class SpanModel(Protocol):
    """The shape the binding asks of a model: the library that runs it, names for its weights,
    the versions of the code that runs them, and ``predict``, each span of a label scored at or
    above ``threshold``, as a ``dict`` of ``start``, ``end``, ``text`` and ``score``."""

    provider: str
    model_id: str
    revision: str
    runtime: Mapping[str, str]

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]: ...
```

A real model rarely has that shape, so wrap it in an adapter that does. Read the model's own
output: offsets and scores often come only when asked for. GLiNER2, for example, gives bare
strings unless asked for spans and confidences. A model's packaging may also leave out what its
runtime needs, such as its tokenizer's library; on Linux, a CPU-only torch wheel avoids a large
GPU download; uv's package index set in a library's own project does not carry to a project
that depends on it, so set it in yours.

The binding is a class whose instance is the async function decide calls:

```python
class SpanBinding:
    """``binding(batch, seconds=s)`` over one loaded model, safe to call concurrently."""

    def __init__(self, model: SpanModel, *, threshold: float = THRESHOLD) -> None:
        self._model = model
        self._threshold = threshold
        self._lock = threading.Lock()  # one inference at a time, across threads and loops
        self._elicitation = f"span extraction, one request per question, threshold {threshold}"
```

Pin the weights: load a model revision by its commit hash, not a branch such as `main`, so
`revision` names the weights that ran. The library and runtime that run them, such as
`gliner2` and `torch`, move the scores too, so `runtime` names their versions: `provider`,
the model, `revision` and `runtime` name what produced the answer, `elicitation` how it was
asked.

### Run inference in a thread, one at a time

Inference blocks, and a binding runs on decide's event loop. Run it in a worker thread, so
the loop and the engine's timeout keep running, and mark a model error on its own ref:

```python
                try:
                    spans = await asyncio.to_thread(self._predict, text, label, deadline)
                except Exception as error:  # never BaseException: a cancellation propagates
                    lines.append(f"{ref}: raised: {type(error).__name__}: {error}")
                    values[ref] = Unavailable(Cause("raised"))
                    requests.append(self._request(label, failed=True))
                    continue
                if spans is None:  # at the call's end: ommat has timed out the whole call
                    lines.append(f"{ref}: timed-out: the model was busy until the call's end")
                    values[ref] = Unavailable(Cause("timed-out"))
                    continue
```

Decide calls a binding concurrently with itself when it is bound under several names or as
`default`, and turns on several loops or threads may share one instance; a model is rarely
safe to share. A `threading.Lock` taken inside the worker thread serialises access across all
of them, where an `asyncio.Lock` belongs to one loop. Keep each call's state in locals, so the
binding is safe to call concurrently:

```python
    def _predict(
        self, text: str, label: str, deadline: float
    ) -> list[tuple[int, int, str, float]] | None:
        """The model's spans as Python values, a torch or numpy scalar made an int or float;
        ``None`` when another inference held the model until ``deadline``."""
        left = deadline - time.monotonic()
        if left <= 0 or not self._lock.acquire(timeout=left):
            return None
        try:
            found = self._model.predict(text, [label], self._threshold)
        finally:
            self._lock.release()
        spans = [(int(s["start"]), int(s["end"]), str(s["text"]), float(s["score"])) for s in found]
        return sorted(spans)
```

The conversion matters: a torch or numpy scalar is not an `int` or `float`, and the answer
must hold Python numbers. A thread cannot be stopped, so when ommat's timeout ends a call,
its inference runs on to its end, holding the lock, and its result is dropped. The wait for
the lock ends with the call, so a call cut while it waits does not go on to run the model,
bar a lock freed in the instant between ommat's end and the binding's own deadline. A `None`
from `_predict` comes at the call's end, when ommat has already timed out the whole call, so
its `timed-out` marker is a guard that is never read.
`decide_sync` and `check` wait for such a thread to end: on Python 3.11 for as long as it
runs, on 3.12 and later for up to 300 seconds, and the process waits for it again at exit.

### Refuse what the model cannot answer

A span model answers an extraction of value type `text` from a text, worded as a label
string. Mark every other ref `unsupported`, each with its reason:

```python
def _refusal(question: RenderedQuestion, image: bool) -> str | None:
    """Why the model cannot answer ``question``, or ``None`` when it can."""
    if question.kind != "extraction":
        return f"the model answers extractions, not a {question.kind}"
    if question.value_type != "text":
        return f"the model reads text spans, not a {question.value_type}"
    if not isinstance(question.wording, str):
        return "the model takes its label as a string, not a structured wording"
    if image:
        return "the source needs an image, which the model cannot read"
    return None
```

A `t:` ref is marked the same way, and so is a source that needs an image, read with
`ommat.bindings.image_need`. A span model gives no label probabilities, so `noul`, `choice`
and `score` are refused alike. A model that gives only `yes` for a `noul` holds `no` as its
own library derives it ([The binding protocol](../spec/README.md#the-binding-protocol)); ommat
ships no such helper for a hand-written binding.

### Check each span

A span must read the value it locates. Check `text[start:end]` against the value the model
returned, and mark the ref `invalid-answer` when they differ:

```python
    for start, end, found, _ in spans:
        if text[start:end] != found:
            lines.append(f"{ref}: invalid-answer: span [{start}, {end}] does not read {found!r}")
            return Unavailable(Cause("invalid-answer"))
```

Offsets are code points, as Python's string indexes are; a model that reports token or byte
offsets needs them converted. Then build the answer through `extraction_value`. A list answer
carries every span and no confidence, since the model gives no one confidence for the list; a
scalar answer takes the best span and its score:

```python
    if question.is_list:  # every span found, and no one confidence for the list
        found_all = [found for _, _, found, _ in spans]
        return extraction_value(found_all, None, None, None, [(s, e) for s, e, _, _ in spans])
    if not spans:  # the model found none: a null value, not a marker
        return extraction_value(None, None, None, None, None)
    start, end, found, score = max(spans, key=lambda s: s[3])
    return extraction_value(found, None, None, score, (start, end))
```

`p_null` is `None`: the model gives no probability that the text holds none. A model that
cuts a long text short answers from part of it: put the ref in the request's `truncated`, or
give a `warnings` line, so the record says so. To learn whether yours cuts, probe it: put a
known entity at the end of texts of growing length, and see where it stops being found. A text
past the window the model was trained on that the model does not cut may read less well: give
it a `warnings` line. A binding that splits the text instead names that in `elicitation`, and
its spans still index the whole text.

### Record the provenance

Each request the call sends is one `RequestProvenance`: the provider, the model, its
revision and runtime, `elicitation`, the method and its threshold, and `prompt`, what the
model received beside the text, which the record already holds as a need, written by
`prompt_text`. The model id and revision are the model's own word, which the binding cannot
check, so it claims them; a request that failed reports no model, its `model_reported`
`not-sent`. Each detail the model cannot give is null, with its reason:

```python
    def _request(self, label: str, *, failed: bool = False) -> RequestProvenance:
        """One model request's provenance: what ran it and how, the arguments it took, and why
        each detail the model cannot give is null; a request that failed reports no model."""
        model = self._model
        return RequestProvenance(
            provider=model.provider,
            model_requested=model.model_id,
            model_reported=None if failed else model.model_id,
            revision=model.revision,
            runtime=dict(model.runtime),
            elicitation=self._elicitation,
            prompt=prompt_text({"labels": [label], "threshold": self._threshold}),
            # The model's own word, not checked.
            claimed=("revision",) if failed else ("model_reported", "revision"),
            attempts=1,
            missing={**NOT_GIVEN, "model_reported": "not-sent"} if failed else NOT_GIVEN,
        )
```

```python
NOT_GIVEN = {
    "request_id": "not-reported",
    "calibrated": "not-reported",
    "usage.input_tokens": "not-reported",
    "usage.output_tokens": "not-applicable",  # an encoder writes no tokens
    "usage.passes": "not-reported",
}
```

`call_result` folds the requests into the call's `CallResult`, each distinct value once.
`precision` is `unrounded`, since the model's scores reach the answer as it computed them,
and `bounded_warnings` keeps the lines within their bounds:

```python
warnings = bounded_warnings(lines)
return call_result(values, requests, precision="unrounded", warnings=warnings)
```

A batch whose every ref is `unsupported` sends no request, so it has no provenance and owes
no reason, as the protocol asks.

## Hold the binding to the protocol

Run the conformance suite before you bind anything, over a binding of either path; `idia_run.py`
runs it over Path A's. Over a leaf that answers only text extractions, most of the suite's
batches come back `unsupported` before any call, so a pass there mostly tests the refusals.
`reports_run.py` runs it over Path B's:

```python
    print("conformance:", check(binding, deterministic=True))
    print("own batches:", check(binding, batches=[OWN], deterministic=True))
```

`ommat.bindings.conformance.check` calls the binding with its own batches, which
[The binding protocol](../spec/README.md#the-binding-protocol) lists, and returns each failure
as a string naming its check; an empty list passes. `deterministic=True` also checks that two
calls with one batch return equal results, which a local model on a CPU should; a model on a
GPU or behind a network may not repeat exactly, so drop the flag there. Pass `batches=` to send batches shaped
like your design's, such as the driver's `OWN`. Call it from synchronous code: it runs its own
event loop and raises `RuntimeError` inside a running one. It is public, but outside ommat's
compatibility promise.

A batch is a tuple of `SourceCall(id, payload, needs, refs, questions)`, its fields in this
order, and each question a `RenderedQuestion(kind, wording, labels, *, descriptions,
value_type, is_list, text, reference_date)`, all but the first three keyword-only; `text` is the `needs` key that holds the
text. The driver's own batch:

```python
OWN = (
    SourceCall(
        "places",
        None,
        {"t:state.text": TEXT},
        (),
        {
            "q:places/street": RenderedQuestion(
                AnswerKind.extraction, "street", (), value_type=ValueType.text, text="t:state.text"
            ),
            "q:places/streets": RenderedQuestion(
                AnswerKind.extraction,
                "street",
                (),
                value_type=ValueType.text,
                is_list=True,
                text="t:state.text",
            ),
        },
    ),
)
```

A binding that passes keeps the protocol; it may still misread its model.

## Read the answer in a rule

`extract(r, ref, null_bar)` returns an extraction's `value`, or `Unsure` with `abstained`
when the value is `None` or `p_null` is given and at or above `null_bar`; a `None` `p_null`
never abstains, and the result never depends on `confidence`. `null_bar` has no default, and
applies only to a model that gives a `p_null`. The list rule uses it:

```python
    "streets": Rule(
        "d:streets",
        t.list_(t.string),
        lambda r: extract(r, "q:places/streets", 0.5),
        version="1",
        reads=["q:places/streets"],
    ),
```

To weigh confidence, read the answer whole. It is an `ExtractionAnswer`, or a marker:

```python
def sure_street(r: ReadView) -> object:
    """The street the model is sure of: the answer read whole, for its confidence."""
    answer = r.read("q:places/street")
    if isinstance(answer, MARKERS):
        return answer
    if answer.value is None or answer.confidence is None or answer.confidence < 0.8:
        return Unsure(Cause("abstained", ["q:places/street"]))
    return answer.value
```

A model's score is the model's own, and need not be calibrated: a bar such as 0.8 is yours to
set by measuring.
A span model's score says how much a span looks like the label, not whether it fills the
question's role: a recipient's name may score as high as a sender's. A score may also shift with
the other labels asked beside it, as [The binding protocol](../spec/README.md#the-binding-protocol)
warns of any question beside others, which is why this binding asks one label per request.
Path A asks a source's questions together in one request, so the two paths' answers to one
design may differ.

A span model's list answer carries no `p_null` and no confidence, so `extract`'s bar never
applies to it: the rule gets the list, and `[]` when the model found none. For a model that
does give a `p_null`, a rule can call `extract` first and then read the answer whole for its
confidence.

## Run it

The driver wraps the model to count its calls, so the run shows that replay makes none; wrap a
real model the same way:

```python
class Counted:
    """A model that counts its calls, so a run can show that replay makes none."""

    def __init__(self, model: SpanModel) -> None:
        self.model = model
        self.provider, self.model_id = model.provider, model.model_id
        self.revision, self.runtime = model.revision, model.runtime
        self.calls = 0

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        self.calls += 1
        return self.model.predict(text, labels, threshold)
```

From `docs/how-to/add_a_source`, run `python reports_run.py`:

```text
conformance: []
own batches: []
inform | Logged for Quay Street. | passed
  answers: {'q:places/street': 'Quay Street', 'q:places/streets': ['Quay Street']}
  replay findings: [] | model calls: 0
inform | Logged for Elm Row, Mill Lane; no one street stands out. | passed
  answers: {'q:places/street': 'Elm Row', 'q:places/streets': ['Elm Row', 'Mill Lane']}
  replay findings: [] | model calls: 0
escalate | the report names no street | passed
  answers: {'q:places/street': None, 'q:places/streets': []}
  replay findings: [] | model calls: 0
```

In the second report the best street scores 0.7, below the rule's bar of 0.8, so `d:street`
is `Unsure` and the plan informs with the list. The third names none: the list is `[]` and the
plan escalates. Each replay rebuilds its plan with no model call.
