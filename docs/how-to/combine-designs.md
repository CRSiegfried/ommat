# Combine small designs into one

This guide writes a workflow selector and a workflow as two small designs, then rolls them up
into one design at run time, so every question either asks on the turn's state goes to the
model in one call. It assumes you have done [Gate a source on an earlier estimate](gate-a-source.md).

The example's files are in `docs/how-to/combine_designs/`: `combine_design.py` holds the designs
and `combine_run.py` the driver, which answers through `ommat.bindings.fixed`, so it runs with
no model. Run it from that folder with `python combine_run.py`, in an environment where ommat is
installed. The spec's [combining designs](../spec/README.md#combining-designs) states the rules;
this guide shows them at work.

## Why the workflow does not wait for the selector

The selector picks a workflow; the email workflow needs to know whether it was picked, and
whether the selector found the message urgent. Gating the workflow's source on the selector,
with a [source requirement](gate-a-source.md), would ask its question a round later, in a call
of its own. Instead the workflow asks its question at once, beside the selector's, and its
rules read the selector's estimates. Rules fire within a round, so the dependency costs no call.

## Declare the ports

A part is a design and its ports: the `d:` refs its rules read that no declaration of it
provides, each with its type. Name a port for what the workflow needs, under its own name, not
the selector's:

```python
EMAIL_PORTS = {"d:email/chosen": t.boolean, "d:email/urgent": YES_NO}
```

The workflow's own policy, `d:email/outcome`, becomes an ordinary rule of the combined design.

## Bind each port and name the top policy

`combine` takes the parts, a provider for each port, and the top policy:

```python
DESIGN = combine(
    Part(SELECTOR),
    EMAIL,
    bind={"d:email/chosen": CHOSEN, "d:email/urgent": "d:urgent"},
    policy=ACT,
    name="assistant",
)
```

A port bound to a `d:` ref, as `d:email/urgent` is, takes that ref's value through a pass-through
table combine builds. A port bound to a rule or table takes that rule's value; here a table turns
the selector's route into the boolean the workflow reads:

```python
CHOSEN = Table("d:email/chosen", t.boolean, "F", ["d:route"], [["email", True], ["Known", False]])
```

The top policy reads the selector's route and the workflow's outcome:

```python
ACT = Table(
    "d:act",
    t.label("pass", "skip", "flag", "file", "ask_folder"),
    "F",
    ["d:route", "d:email/outcome"],
    [["email", "-", Input("d:email/outcome")], ["other", "-", "pass"]],
)
```

## What a turn asks

Both parts' sources name no oracle and need only the state, so the turn asks them in one call,
which the idia binding sends as one request. The driver runs two turns and replays each:

```text
'File the invoice under Billing': calls [['email/ask', 'router']]; decision file; findings []
'What is the weather like?': calls [['email/ask', 'router']]; decision pass; findings []
email alone, chosen and not urgent: decision ask_folder
email alone, chosen this turn: decision ask_folder
```

The workflow's question is asked on the second turn too, though the selector picks `other`:
that is the price of asking every question at once. What else to put in the first round
follows from what each source waits for:

- A question: never gate it on the selector, which would ask it a round later in a call of its
  own.
- A read that needs nothing but the state or the account, such as listing the mail folders:
  leave it ungated, so it runs in round 1 and anything grounded on it is asked in round 2.
  Every turn then makes the call, and its result is in every record.
- A read whose arguments come from round-1 answers: gate it on the port. It waits for those
  answers anyway, and the port is decided in the same round, so the gate costs no round and
  saves the call on another workflow's turn.
- A write: always gate it on a decided value and on the port, under an oracle of its own,
  needing only the call it makes ([A source that changes state](add-a-source.md#a-source-that-changes-state)).

## Test a part alone

`standalone` gives a part a constant for each port, as a table with no input, so a test loads
and decides the workflow with no selector:

```python
    alone = load(standalone(EMAIL, {"d:email/chosen": True, "d:email/urgent": "no"}))
```

To run a part alone in production, with a port's value set each turn rather than fixed in the
design, name it in `given`. `standalone` adds the source `ports`, under the oracle `ports`,
providing `t:email/chosen` for `d:email/chosen`, and you bind it at decide:

```python
    given = load(standalone(EMAIL, {"d:email/urgent": "no"}, given=["d:email/chosen"]))
    ports = fixed({"t:email/chosen": True})
    decision = decide_sync(given, {"message": "Move it"}, {"default": model, "ports": ports})
```

The record holds the value as the source's tool result, so the turn replays.

## What combine refuses

`combine` raises `InvalidCombination`, listing every failure, for parts whose states differ; an
id or a parameter path two parts declare; a type, group, target or class tree two parts declare
differently; a port some part provides; a port left unbound; and a bind key that is no port. A
type, group, target or class tree two parts declare alike is kept once, so each part keeps what
it reads. A parameter is never shared: give each part its own, as `email/bar` here, so tuning one
part never moves another. It renames nothing, so give each part's ids and refs a prefix of its
own, as `email/` here. Everything else, such as a read with no provider, is load's to refuse.

An oracle name is shared across parts as it is within one design: the sources under it in a
round are one call. A binding under a name several parts use answers every part's sources
under it, so a `context` oracle that gives one part an account and another a date is one
`fixed` holding both values.
