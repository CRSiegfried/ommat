# Lay out a design module so it reads

A design is code a person or an agent will read long after it was written, to find what a
turn decided and why, or to change one bar. This guide lays out one design module so a reader
can go top to bottom and find any node without tracing builders. It covers how the code reads;
what ommat requires of a design is in the [spec](../spec/README.md) and the
[ommat skill](../../skills/ommat/SKILL.md). It assumes you have done
[Write your first design](../tutorials/write-a-design.md).

The example's files are in `docs/how-to/lay_out_a_design/`: `refund_design.py` holds a refund
desk's design laid out as this guide says, `refund_checks.py` two checks its tests can hold,
and `refund_run.py` the driver, which answers through `ommat.bindings.fixed`, so it runs with
no model. Run it from that folder with `python refund_run.py`.

These are practices, not rules: where one fights the design in front of you, do what reads
best and say why in a comment.

## Open with the docstring as the map

A design module opens with a docstring a reader can use without opening anything else: what it
decides and its outcomes, then its sources, then its rules in graph order, each ref with what
its value means.

```python
"""The refund desk as an ommat design: what to do with a customer's message about an order.

It decides ``refund``, ``store_credit``, ``decline`` or ``answer``, or stays ``Unsure`` and
hands the message to a person.

Sources:

- ``ask`` (the model): ``q:refund``, ``q:reason``.
- ``orders`` (the order system): ``t:order_age``, the order's age in days.

Rules, in graph order:

- ``d:refund``: ``yes`` when the customer asks for their money back, ``no`` when not.
- ``d:reason``: why, ``damaged``, ``late`` or ``changed_mind``.
- ``d:within_window``: whether the order is young enough to refund.
- ``d:outcome``: the decision.

Bars are set in ``BARS``; the wording is in ``QUESTIONS``.
"""
```

Give each source's oracle and what gates it, and write a family of refs once with its slot,
`d:prefetch_<op>`. A module that is one part of a [combined design](combine-designs.md) also
lists the ports it reads and which part provides them. `python -m ommat.outline` prints the
nodes in the same graph order, so the map can be checked against it.

A map that disagrees with the code is worse than none, so hold it in a test. `map_disagrees`
compares the refs the docstring names with those the design provides:

```python
def map_disagrees(module: ModuleType, design: Design) -> dict[str, list[str]]:
    """The refs the module's docstring names that the design does not provide, and those it
    provides that the docstring leaves out; both empty when the map is true."""
    named = set(MAP_REF.findall(module.__doc__ or ""))
    outline = design_outline(load(design).document)
    provided = {ref for node in outline["nodes"] if node["kind"] != "state" for ref in node["refs"]}
    return {"not provided": sorted(named - provided), "not in the map": sorted(provided - named)}
```

## Keep the sections in one order

Top to bottom: constants and labels, the questions, the types, the rule functions in graph
order, the bars, the tables, the design, reading the record back, and running it. A one-line
comment heads each section, so a reader scrolling knows where they are.

## Name each rule function for the ref it provides

`d:within_window` is computed by `within_window`, and its docstring starts with the ref:

```python
def within_window(r: ReadView) -> Any:
    """``d:within_window``: whether the order is at most ``bars/window_days`` old."""
    held = determined(r, "t:order_age")
    if not isinstance(held, tuple):
        return held  # a marker passes on, with its cause
    (age,) = held
    return age <= r.param("bars/window_days")
```

Define it at module level, never as a lambda or closure: a record replays, and a stored design
reloads, by import, and the outline shows the function's real name. Where one rule per value
does the same work, write a small named function per value calling one shared `_`-prefixed
helper. Keep a rule's `reads` short enough to check against its body at a glance: load refuses
a read nobody provides, but nothing tells a reader whether the list is complete.

## Write refs as literals

Write `"d:refund"` where it is used, not a constant. Load refuses a design that reads a ref
nobody provides, so a typo fails at load, and literals keep the graph greppable. Keep
constants for the names a caller binds, such as an oracle's:

```python
ORDERS = "orders"
"""The oracle name the caller binds to the order system."""
```

Refs built from a mapping of question ids, `f"q:{qid}"`, are fine: writing them out would
repeat the wording's keys.

## Name a table's inputs

A table's rows are lists, one cell per input in order, so a row written by position breaks when
an input is added and cannot be read without counting. Name the inputs once, and build each
row by name:

```python
OUTCOME_INPUTS = {
    "refund": "d:refund",
    "within_window": "d:within_window",
    "reason": "d:reason",
}


def row(out: object, **cells: object) -> list[object]:
    """One row of ``d:outcome``, each cell named by its input, ``-`` where none is given."""
    unknown = cells.keys() - OUTCOME_INPUTS.keys()
    if unknown:
        raise KeyError(f"d:outcome has no input named {sorted(unknown)}")
    return [cells.get(name, "-") for name in OUTCOME_INPUTS] + [out]
```

Say the hit policy, and under `F`, first match wins, order the rows as the decision reads, with
the guard rows for markers first and a comment on why:

```python
OUTCOME = Table(
    "d:outcome",
    OUTCOME_T,
    "F",  # the first matching row wins
    list(OUTCOME_INPUTS.values()),
    [
        # A refund we could not settle goes to a person before any row reads its value.
        row(Unsure, refund="Undetermined"),
        row("answer", refund="no"),
        row("decline", refund="yes", within_window=False),
        row("refund", refund="yes", within_window=True, reason="{damaged, late}"),
        row("store_credit", refund="yes", within_window=True, reason="changed_mind"),
        row(Unsure),
    ],
)
```

Prefer a table when a decision is a short list of conditions over a few values, and a function
when it computes a value. The spec's [decision tables](../spec/README.md#decision-tables) give
every cell's meaning; note that `-` matches a marker, `Unasked` included, and `Known` does not.

## Give each decision its own bar, and say where it comes from

Every number a rule compares against is a `Parameter` read with `r.param` and declared in the
rule's `params`, never a closure capture, so the design hash, the lints and `switching_values`
see it. Name bars by the decision they serve, one per decision, so raising one moves only that
decision, and say beside each where its value comes from:

```python
BARS = {
    # Each bar is one decision's, so raising one moves only that decision.
    "bars/refund_at": Parameter(t.number(0, 1), 0.7),
    "bars/refund_lead": Parameter(t.number(0, 1), 0.2),
    "bars/reason_at": Parameter(t.number(0, 1), 0.6),
    # Starts at the refund's lead; it is its own bar so it can move alone.
    "bars/reason_lead": Parameter(t.number(0, 1), 0.2),
    # The shop's policy, not a measurement: thirty days from the order.
    "bars/window_days": Parameter(t.integer(0, 365), 30),
}
```

A bar set from a measurement should name the measurement, so a reader knows to measure again
before moving it.

## Make each decision once

A decision is made by one rule or table, and every rule that needs it reads its ref. A rule that
recomputes it from the answers or from module constants is a second copy that can drift, and a
reader finds only one of them. A rule that only checks whether its inputs are available may do
so itself; say so in its docstring.

## Bump a rule's version when what it calls changes

The design hash covers the declared parts, the parameters and each rule function's own source.
It cannot see what a rule calls or imports: a helper in another module, a constant, a date
library. Change the rule's `version` when one of those changes behaviour, so an old record
replays as `design-mismatch` rather than agreeing with code it never ran.

## Read the record back into your own type

The caller should not handle ommat's types. Read the entries once, since `record.entries` is
decoded afresh at each read, and return a frozen type of your own:

```python
def read_refund(record: Record) -> RefundReading:
    """The turn's outcome and reason; ``outcome`` is ``None`` when a person must look, and
    ``defect`` says the design, not the evidence, is at fault (the escalation view's test)."""
    entries = record.entries  # decoded afresh at each read, so read once
    held = escalation(record)
    return RefundReading(
        outcome=_value(entries, "d:outcome"),
        reason=_value(entries, "d:reason"),
        defect=held is not None and held["defect"],
    )
```

An undecided turn is not an error: raise only on `defect`, a rule or the codec at fault, and
hand the rest to whoever decides when the model could not.

## Use one word per meaning, and no history

Define a design's own terms once, in its map or a shared module, and use each for one thing.
Leave plans, ticket numbers and "legacy" out of the code; git holds the history. Do cite the
evidence behind a measured choice, such as a measurement's id, so a reader knows where to look
before changing it.

## Check that a tidy moved nothing

Renaming or moving rule functions changes every design hash, since each function's module,
qualname and source are in it, and should change nothing else. Compare the two designs with
those three dropped:

```python
def moved_nothing(before: Design, after: Design) -> bool:
    """Whether two designs differ only in where their rule functions live and what they are
    called: each function rule's module, qualname and source dropped, the documents are equal."""
    return _without_places(before) == _without_places(after)
```

Then replay a few stored records under the new design: each finding should be
`design-mismatch` and nothing else.

## Run it

```text
'The mug arrived broken, I want my money back': RefundReading(outcome='refund', reason='damaged', defect=False)
'I changed my mind about the lamp, refund please': RefundReading(outcome='store_credit', reason='changed_mind', defect=False)
'The chair from March is late, refund it': RefundReading(outcome='decline', reason='late', defect=False)
'Not sure the vase is what I wanted': RefundReading(outcome=None, reason='changed_mind', defect=False)
map disagrees: {'not provided': [], 'not in the map': []}
```

The last turn's model was torn on whether the customer wants a refund, so `d:refund` is
`Unsure`, the guard row sends the turn to a person, and `outcome` is `None`.
