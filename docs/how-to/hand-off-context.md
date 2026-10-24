# Hand off what a turn settled

This guide gives the next model in a conversation what ommat settled in a turn: the plan, its
steps and the estimates you choose, worded as you declare them. It assumes you have done
[Write your first design](../tutorials/write-a-design.md). The spec's
[context block](../spec/README.md#the-context-block),
[handoff view](../spec/README.md#the-handoff-view) and
[conversation renderers](../spec/README.md#the-conversation-renderers) state the rules; the
guide shows them at work.

The handoff is not an audit trail. It carries what the next model needs to carry on, and
never the guard, a requirement, a probability or provenance; the record keeps those.

The example's files are in `docs/how-to/hand_off_context/`: `support_design.py` is the design
and `handoff_run.py` the driver, which answers with a stand-in binding, so it runs with no
model. Run it from that folder with `python handoff_run.py`, in an environment where ommat is
installed.

## Declare the context block

A design hands nothing off until it declares a context block. Each entry flags one ref,
`t:state` or a `d:` ref, with or without a field path, and words it with a `Text` whose slots
name refs or parameters. One slot names the flagged ref:

```python
CONTEXT = Context(
    [
        ContextEntry(
            "d:refund",
            Text("Refund requested: {refund}", [{"name": "refund", "origin": {"ref": "d:refund"}}]),
        ),
        ContextEntry(
            "t:state.order.status",
            Text(
                "Order {id} is {status}",
                [
                    {"name": "id", "origin": {"ref": "t:state.order.id"}},
                    {"name": "status", "origin": {"ref": "t:state.order.status"}},
                ],
            ),
        ),
    ]
)
```

Compile renders each entry before the policy runs and writes the result into the record's
`plan` item, so replay checks the words byte for byte and `redact` digests them.

The design takes the block as `context`. Load checks each entry: it refuses a `q:` ref, two
entries flagging one ref, a text with no slot naming its flagged ref, and any slot over a field
the design classes `identifier`, such as the customer's email:

```python
DESIGN = Design(
    name="support",
    state=STATE,
    sources={"ask": ASK},
    rules={
        "refund": Rule("d:refund", t.label("no", "yes"), refund, version="1", reads=["q:refund"])
    },
    policy=Policy(policy, version="1", reads=["d:refund", "t:state.order.id"]),
    tools=["refund_order"],
    classes={"t:state": {"email": "identifier"}},
    context=CONTEXT,
)
```

## Hand it to the next model

`handoff(record)` reads the record and returns what the block switches on, and
`ommat.conversation.to_messages` renders it as one message. The driver decides a turn where
the customer clearly asks for a refund:

```python
    decision = decide_sync(loaded, STATE, {"default": answering(0.9)})
    (message,) = to_messages(handoff(decision.record), role="user")
    print(message["content"])
```

`role` has no default. An entry's text is the input's own: here the order comes from the
caller's state, and a slot over the customer's message would put their words in the handoff
as written. Use `role="user"`, or fence the content as data, when a slot reads free text, and
`"system"` or `"developer"` only when every slot reads values your own code settled.

The plan's steps appear only when the guard passed it, each argument tagged as a literal, a
board ref with its value, or a binding to an earlier step's result, so the next model never
sees steps that will not run.

## Run the steps as tool calls

`to_tool_calls(decision)` renders a passed `execute` plan as OpenAI-shape tool calls, each id
the step's idempotency key, the same key `ommat.helpers.run_plan` passes:

```python
    for call in to_tool_calls(decision):
        print(call["function"])
        print(call["id"] == f"{decision.record.record_hash}:1")
```

The calls end before the first step that binds an earlier step's result, since its argument
exists only once that step has run. Run such a plan with `run_plan`; a tool that keeps its
idempotency keys runs no step twice.

## When an estimate is not settled

The driver's second turn answers too close to call. The rule abstains, the policy asks to
clarify, and the entry over `d:refund` says so instead of rendering a value:

```python
    unsure = decide_sync(loaded, STATE, {"default": answering(0.6)})
    (message,) = to_messages(handoff(unsure.record), role="user")
    print(message["content"])
```

The engine never words a marker. An entry whose slot reads one holds the marker, and
`to_messages` writes its ref, kind and cause.

The driver prints both turns' handoffs and the tool call:

```text
Plan: execute
Reason: the customer asked for a refund
Step 1: refund_order {"order":{"ref":"t:state.order.id","value":"A-1001"}}
Refund requested: yes
Order A-1001 is delivered
Decided by design support (sha256:b4c10170984a81abc3bd88c5f7192120e135a178bd9e31805161bdc4dfa5ac16); record sha256:5013f8d89a2ec632c8f1431cedfa581a370b57c708603b24b6153b908bfe7c2d
{'name': 'refund_order', 'arguments': '{"order":"A-1001"}'}
True
Plan: clarify
Reply: Would you like a refund for this order?
d:refund: not settled (unsure: abstained)
Order A-1001 is delivered
Decided by design support (sha256:b4c10170984a81abc3bd88c5f7192120e135a178bd9e31805161bdc4dfa5ac16); record sha256:c34897f04fd5b52643c488c6daddb172f5a1e3f2f0fff4f7a859e97de7329b85
```

## Switch parts off

Every part the block includes by default has a switch. `Context(entries, plan=False)` leaves
out the plan's kind and text, `steps=False` its steps, and `record=False` the design and record
hash; the entries are always given. With no block, `handoff` returns `None` and nothing is
handed off.
