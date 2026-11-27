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

A part's rules stay ordinary rules of the combined design, `d:email/outcome` among them.

## Bind each port and add the rule that reads across parts

`combine` takes the parts and a provider for each port. A rule that reads more than one part's
estimates goes in as a part of its own, a design that holds only that rule:

```python
DESIGN = combine(
    Part(SELECTOR),
    EMAIL,
    Part(Design(STATE, rules={"act": ACT})),
    bind={"d:email/chosen": CHOSEN, "d:email/urgent": "d:urgent"},
    name="assistant",
)
```

A port bound to a `d:` ref, as `d:email/urgent` is, takes that ref's value through a pass-through
table combine builds. A port bound to a rule or table takes that rule's value; here a table turns
the selector's route into the boolean the workflow reads:

```python
CHOSEN = Table("d:email/chosen", t.boolean, "F", ["d:route"], [["email", True], ["Known", False]])
```

`d:act`, the entry the caller acts on, reads the selector's route and the workflow's outcome:

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
that is the price of asking every question at once.

Interpret asks a source in the first round after everything it depends on is on the board:
its needs, its requirement's ref and, for a grounded source, its candidate set. So give each
source every dependency it really has and no others: one it lacks only delays it, and a need
it lacks is also sent to its binding. A source that depends on nothing but the state is asked
in round 1. A call that should happen only when decided depends on that decision: a source
requirement on the one estimate that says to make it, with every other condition, such as a
combined part's port, folded into that estimate, and needs that are only the call's
arguments, as a rule writes them. A need alone does not hold a source back once it holds a
value, `null` included. Every write is such a call, and so is a read that is costly,
rate-limited, sensitive or has a side effect.

So in a combined design:

- A question depends on the state alone. Gated on the selector, it would be asked a round
  later in a call of its own.
- A read that needs nothing but the state or the account, such as listing the mail folders,
  has no other dependency, so it runs in round 1 and anything grounded on it is asked in round
  2. Every turn then makes the call, and its result is in every record.
- A read whose arguments come from round-1 answers needs them, and should happen only on its
  workflow's turn, so its requirement's estimate reads the port too. Both are decided in round
  1, so it runs in round 2 either way.
- A write depends on the decision to make it, the port folded in, under an oracle of its own
  ([A source that changes state](add-a-source.md#a-source-that-changes-state)).

## When parts were measured on different states

A combined design has one state, but a part's questions may have been worded and their bars
measured on another shape: a workflow measured on `{"request": ...}` beside a selector
measured on `{"earlier_turns": [...], "current_turn": ...}`. Under the idia binding, the state
a source needs is what the model reads, so asking that part's questions on the combined state
moves their answers. Keep the shape each part was measured on instead:

- Add a rule that builds the part's view from the state alone, such as `d:email/view`, whose
  value is `{"request": current_turn}`. A rule over the state alone fires in round 0, so
  a source that needs it is still asked in round 1.
- Give the part's questions `needs=["d:email/view"]` in place of `t:state`, under an oracle
  of their own, such as `view_request` beside the selector's `selector`.
- Bind that oracle to its own binding over the same model, naming the view as its state:
  `oracle_binding(provider, state="d:email/view")`, the selector's staying at
  `state="t:state"`. The binding sends the view's value unwrapped, so the model reads exactly
  the measured shape.

Round 1 then makes two calls, each one request, run concurrently, so the turn waits for the
slower; the round's rules fire once both return, and one failing leaves the other's answers
standing. The cost is the packing between the two parts. Replay recomputes the view. A
source that builds the view as a tool result would instead push the part's questions to
round 2.

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
id one part declares as a source and another as a rule; a parameter path two parts declare; a
source, rule, type, group or target two parts declare differently under one id; a port some part
provides; a port left unbound; and a bind key that is no port. A source, rule, type, group or
target two parts declare alike is kept once, so each part keeps what it reads and still runs
alone. A question several parts ask alike goes in a source of its own, declared alike in each
part with the rules its needs come from, and is asked once. A parameter is never shared: give each part its own, as `email/bar` here, so tuning one
part never moves another. It renames nothing, so give each part's ids and refs a prefix of its
own, as `email/` here. Everything else, such as a read with no provider, is load's to refuse.

An oracle name is shared across parts as it is within one design: the sources under it in a
round are one call. A binding under a name several parts use answers every part's sources
under it, so a `context` oracle that gives one part an account and another a date is one
`fixed` holding both values.
