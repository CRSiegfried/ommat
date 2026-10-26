# Gate a source on an earlier estimate

This guide asks a later question only when an earlier estimate holds a value, such as asking
whether a message states an amount only when its intent is a refund, without sending that
estimate to the model. It assumes you have done [Add a source](add-a-source.md).

The example's files are in `docs/how-to/gate_a_source/`: `gate_design.py` is the design and
`gate_run.py` the driver, which answers with a stand-in binding, so it runs with no model. Run
it from that folder with `python gate_run.py`, in an environment where ommat is installed.

## Why not a need or a `when`

Two ways that look right do not work:

- **A need.** A source's needs are what its binding sends. Making `d:intent` a need sends the
  intent to the model beside the state. Under `oracle_binding(provider, state="t:state")` a
  source whose need sits beside the state is refused with `unsupported`, since `state=` sends
  one need unwrapped ([the idia binding](../spec/README.md#the-idia-binding)). Without
  `state=`, it changes what the model reads.
- **A function `when`.** A `when` reads only its source's needs and the state, so reading
  `d:intent` there is an undeclared read, and the source's refs become `Unavailable`.

A source requirement is the way: it names a `d:` ref and the values that let the source be
asked. The ref orders the source after its rule, as a need would, but nothing sends it.

## Project the value the gate tests

A source requirement tests a whole `d:` ref against a list of values, by canonical bytes, so a
field of a record estimate cannot be its ref. When the value you gate on is a field, give it a
rule of its own that projects it. Give that rule a label type: load then refuses a requirement
value the label type cannot hold, where a plain `string` would let a misspelt value load and
never match.

```python
def intent_label(r: ReadView) -> object:
    """The intent alone, as a label a source requirement can test."""
    return r.read("d:triage.intent")
```

```python
    "intent_label": Rule(
        "d:intent_label", t.label(*INTENTS), intent_label, version="1", reads=["d:triage.intent"]
    ),
```

A marker read passes through the rule unchanged, so the gate sees the triage's own reason when
the intent is not settled.

## Declare the requirement

Put the requirement on the source to gate. Its needs stay what the model should read:

```python
REFUND = Source(
    {"q:refund/amount_stated": Question("noul", Text("Does the message state an amount?"))},
    needs=["t:state.message"],
    requirement=SourceRequirement("d:intent_label", ["refund"]),
)
```

A requirement gates every question of its source. When only some questions should wait, put
those in a source of their own and gate that one. A source may have both a requirement and a
function `when`: both must hold, the requirement tested first.

## What the gated source gets

The requirement's ref counts as a need when its entry is a marker, so what the gated source
gets depends on what the gate holds:

| The gate holds | The gated source's refs |
|---|---|
| one of the values | asked |
| another value | `Unasked` with `when-false` naming the gate |
| `Unsure` or `Unasked` | `Unasked` with `need-undetermined` naming it |
| `Unavailable` | `Unavailable` with `need-unavailable` naming it |

The driver runs one turn for each of the first three:

```text
'I want my money back, 40 euros': gate refund; refund source asked; plan inform
'Why was I charged twice?': gate billing; refund source Unasked (when-false: d:intent_label); plan inform
'Hello?': gate Unsure; refund source Unasked (need-undetermined: d:intent_label); plan escalate
```

Read the gated estimate for a value only on the branch where the gate held, so a turn the gate
closed is not taken for a missing answer:

```python
    if intent != "refund":
        return Plan("inform", reply="Noted.")
    # Only a refund asked the gated question, so only here is its estimate read for a value.
    if r.read("d:amount_stated") == "yes":
        return Plan("inform", reply="A refund with its amount: queued for review.")
```

## What load refuses

- A requirement on a `q:` answer or on `t:state`: route it through a rule.
- A requirement on a field path, such as `d:triage.intent`: project the field, as above.
- A value the ref's type cannot hold, a value given twice, or a ref with no provider.
- A cycle, such as the gated source's answer feeding the rule it is gated on.
