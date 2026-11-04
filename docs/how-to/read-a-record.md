# Read a record

This guide answers the questions a caller asks of a turn's record: what an entry holds, which
call answered a question, why a turn is undecided, which rule failed on a turn that decided, and
what the model answered across many turns. It assumes you have done
[Write your first design](../tutorials/write-a-design.md). The spec's
[record document](../spec/README.md#the-record-document),
[escalation view](../spec/README.md#the-escalation-view),
[defects view](../spec/README.md#the-defects-view) and
[answers table](../spec/README.md#the-answers-table) state the rules; the guide shows them at
work.

A record is its log, a list of JSON items, and every question below is a lookup by ref or one
of ommat's views over it. Read it as data, through `log`, `entries` and `unhashed`, and use a
view where it applies a rule you would otherwise copy.

The example's files are in `docs/how-to/read_a_record/`: `desk_design.py` is the design, which
asks a model whether the customer wants a refund and reads the date from a clock oracle, and
`desk_run.py` the driver. Run it from that folder with `python desk_run.py`, in an environment
where ommat is installed. It needs no model: `ommat.bindings.fixed` answers each asked ref from
a mapping, so it stands in for the model and binds the clock to a date the caller holds:

```python
LOADED = load(DESIGN)
CLOCK = fixed({"t:today": "2026-11-04"})


def turn(message: str, ordered: str, refund: dict[str, float] | None) -> Decision:
    """One turn, the model answering ``refund``, or not answering when it is ``None``."""
    model = fixed({} if refund is None else {"q:refund": refund})
    state = {"message": message, "ordered": ordered}
    return decide_sync(LOADED, state, {"default": model, "clock": CLOCK})
```

A ref `fixed` does not hold comes back `unsupported`, as the third turn's refund question does.

## Read an entry

`record.entries` maps each ref to its entry item, a grounded ref as written. An entry holds a
`value` or a `marker`, never both:

```python
def held(entry: dict[str, Any]) -> object:
    """An entry's value, or its marker's kind and cause code."""
    if "value" in entry:
        return entry["value"]
    return f"{entry['marker']['kind']} {entry['marker']['cause']['code']}"
```

Use `record.entries.get(ref)` where a ref may hold no entry, such as a ref a cancelled turn
never reached. Look entries up by ref, never by position: the log's order follows the
turn's rounds, which a design change can move.

## Find the call that answered

A `q:` or `t:` entry's `call` is the log index of the call item that wrote it, and `null` for a
rule's entry or `t:state`. The call item holds the sources it asked, each question's kind and
identity, and the provenance the binding gave, `provider`, `model_reported` and `prompt` among
it:

```python
    asked = record.entries["q:refund"]
    call = record.log[int(asked["call"])]
```

`fixed` reports no provenance, so the example's `provider` is `None`.

## Route an undecided turn

When the decision is a marker, `escalation(record)` gives its kind and cause, the root cause
found by following `required` causes to the entry they name, whether that root cause is a
defect, and the record hash. It is `None` for a turn that decided:

```python
    escalated = escalation(silent.record)
    assert escalated is not None
    print("undecided", escalated["kind"], escalated["root_cause"]["code"])
```

## Find a rule that failed on a decided turn

A policy that gives a value hides a rule that raised beside it: the escalation is `None`.
`defects(record)` lists every entry whose cause is a defect by the escalation's test, whatever
the policy gave. In the second turn the order date does not parse, so `d:days_open` raised
while the turn still decided. The view is `None` for a record written before format 8:

```python
    for found in defects(late.record) or []:
        print("defect", found["ref"], held(found), "estimate", late.estimate)
    print("tracebacks at", sorted(late.record.unhashed["exceptions"]))
```

The traceback is in `unhashed["exceptions"]`, keyed by the entry's log index as a string, as
JSON keys are. Route on the views' test rather than a copy of it, so a new defect rule reaches
your code.

## Read the answers across turns

`ommat.evaluation.answers(records, loaded)` gives one row per question a call answered in each
record, per candidate for a per-candidate question: the answer or marker, its `top` label, the
question's identity and targets, and with a gold set the gold answer:

```python
    for row in answers([sure.record, late.record, silent.record], LOADED):
        marker = row["marker"]
        print("answer", row["ref"], row["top"] if marker is None else marker["cause"]["code"])
```

The driver prints:

```text
d:days_open 5
q:refund {'yes': 0.9, 'no': 0.1} from ['ask'] None
defect d:days_open Unavailable raised estimate refund_order
tracebacks at ['7']
undecided Unavailable unsupported
answer q:refund yes
answer q:refund yes
answer q:refund unsupported
```

## Look around a record

For questions none of these answers, `record.document` is the record's canonical JSON, which
`jq` reads as it stands. To see a turn's rounds, calls and each value's lineage, write the
document to a `.json` file with `loaded.document` beside it and run
`python -m ommat.lens <file> --open`. The record holds no read set: what a rule read is in the
design, which is why the lens takes both.
