# Hand off what a turn settled

This guide gives the next model in a conversation what ommat settled in a turn, two ways: the
handoff, the plan and the estimates you choose, worded as you declare them; and the
transcript, every call the turn made and its conclusion, as tool calls and their results. It
assumes you have done [Write your first design](../tutorials/write-a-design.md). The spec's
[context block](../spec/README.md#the-context-block),
[handoff view](../spec/README.md#the-handoff-view),
[transcript view](../spec/README.md#the-transcript-view) and
[conversation renderers](../spec/README.md#the-conversation-renderers) state the rules; the
guide shows them at work.

Neither is an audit trail, and neither renderer says a word of its own: beyond fixed labels,
each sentence it writes is a plan's `reason` or `reply` or a text you declared. The transcript
also carries the turn's values as recorded, as the section on its calls says. The
handoff holds no guard account; the transcript holds the verdict's failures and each step's
requirements. Only the record keeps the provenance and the prompts.

The example's files are in `docs/how-to/hand_off_context/`: `support_design.py` is the design,
`handoff_run.py` the driver, which answers with a stand-in binding, so it runs with no model,
and `agent_run.py` a Pydantic AI agent driven by a stand-in function. Run each from that folder
with `python handoff_run.py`, in an environment where ommat is installed; `agent_run.py` needs
`ommat[pydantic-ai]`.

## Choose what to hand off

The two overlap: the handoff's plan and entries are also in the transcript's decide result.
For an LLM that carries on the conversation, give it the transcript: it sees the calls as its
own tool calls, and the steps it should run as calls it made. Use the handoff when the next
reader takes instructions rather than a tool history, or beside the transcript as short
instructions: declare the block with `plan=False`, `steps=False` and `record=False`, so it adds
only your entries' words to what the transcript already holds.

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
entries flagging one ref, a text with no slot naming its flagged ref, a text with nothing but
whitespace outside its placeholders, such as `"{refund}"` alone, so each entry says what its
value is, and any slot over a field the design classes `identifier`, such as the customer's
email:

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
`ommat.conversation.handoff_text` gives it as labelled lines. In the driver, `STATE` is the
turn's state, a dict of the customer's message, email and order, and `answering(p)` is a
stand-in binding that answers the refund question `yes` with probability `p`. It decides a
turn where the customer clearly asks for a refund:

```python
    decision = decide_sync(loaded, STATE, {"default": answering(0.9)})
    print(handoff_text(handoff(decision.record)))
```

Place the text where your framework takes per-run guidance: Pydantic AI's `instructions`, or a
system or developer message. An entry's text is the input's own: here the order comes from the
caller's state, and a slot over the customer's message would put their words in the handoff as
written. A step's argument shows its value too, so a step that passes the customer's message or
email shows it, and load's `identifier` refusal covers slots alone. When a slot or an argument
reads free text, fence the text as data in a user message instead.

The plan's steps appear only when the guard passed it, each argument tagged as a literal, a
board ref with its value, or a binding to an earlier step's result. `Verdict:` gives the guard's
kind, so an `execute` plan the guard refused shows `Verdict: refused` and no steps.

## Give the model the turn's calls

`transcript(record)` gives every source call of the turn and its conclusion, unworded:
each call's refs and answers as recorded, the board's estimates, the plan, the verdict and the
context entries, and the steps the caller still has to run. To the next model a source is a
tool, so `to_chat_completions` renders each call as an assistant tool call and its `tool`
result, then a `decide` call whose result is the turn's conclusion, then the pending steps as
calls with no result:

```python
    for message in to_chat_completions(transcript(decision.record)):
        if message["role"] == "assistant":
            for call in message["tool_calls"]:
                print("call", call["id"], call["function"]["name"], call["function"]["arguments"])
        else:
            print("result", message["tool_call_id"], len(message["content"]), "bytes")
```

Append the messages after your own user message for the turn. Each id comes from the record
hash, so one record always renders to the same messages, byte for byte. The same design, state
and answers give the same record: when a turn can repeat one exactly, as a retry does, pass
each rendering its own `turn`, such as the turn's number, `to_chat_completions(view, turn=3)`,
so no tool-call id appears twice in the conversation, which a provider may refuse. A `turn` is
exactly a `str` or an `int`, an `int` within 2^53 - 1 either side of zero and a `str` with no
lone surrogate, so a nanosecond clock is too large; anything else raises `ValueError`. A source's tool name
is its name with each `/` as `-`. A step's tool name is its name in the design, which must be
the name of your tool: the renderers raise `ValueError` for one a provider would refuse.

The transcript gives every answer, estimate and argument as the record holds it, the customer's
words and any `identifier` field among them: load's `identifier` refusal covers a context
entry's slots, not the transcript. Where a turn reads such values, hand off through the handoff
alone, or share the redacted view instead.

The last message calls the plan's steps, and your code runs them and appends their results. A
tool run this way gets no idempotency key. When a repeated call would do harm, run the plan with
`ommat.helpers.run_plan(decision, tools)` instead: it takes the `Decision` and a mapping from
each tool name to an async callable that takes the step's arguments and an `idempotency_key`
keyword, and returns `{"results": [[n, result], ...], "raised": ...}`. Answer each pending call
`..._s<n>` with step n's result. A plan with a step that binds an earlier step's result leaves
no pending calls, so run it with `run_plan` too and tell the model what ran in your own message.

When the plan is `clarify` or `inform`, the turn ends in decide's result, and the reply is the
design's own words. Send it to the user and append it as the assistant's message, so the next
turn's history holds what was said; no model call words it again. An `escalate` plan runs no
step either, and the handoff gives its `Reason:`. A `broken` plan has none, and the handoff
holds no cause, so `Plan: broken` stands beside `Verdict: passed`: the transcript's decide
result holds the cause, so hand that off, or word the outcome in your own code.

## When an estimate is not settled

`handoff_run.py`'s second turn answers too close to call. The rule abstains, the policy asks to
clarify, and the entry over `d:refund` gives its marker's kind and cause instead of a value:

```python
    unsure = decide_sync(loaded, STATE, {"default": answering(0.6)})
    print(handoff_text(handoff(unsure.record)))
```

`handoff_run.py` prints both turns' handoffs and the first turn's calls:

```text
Plan: execute
Verdict: passed
Reason: the customer asked for a refund
Step 1: refund_order {"order":{"ref":"t:state.order.id","value":"A-1001"}}
Refund requested: yes
Order A-1001 is delivered
Design: support sha256:bb5f7c6c348ad87ea480b1868416e93624aa9e39c38a0c990ed67c88342c4df5
Record: sha256:d380bfe188f37d82e59eb0803fdfd645ff769c4390013e4f19ea8ab31e09afad
call om_d380bfe188f37d82e59eb080_c1 ask {"asks":["q:refund"]}
result om_d380bfe188f37d82e59eb080_c1 97 bytes
call om_d380bfe188f37d82e59eb080_d decide {"design":{"name":"support","hash":"sha256:bb5f7c6c348ad87ea480b1868416e93624aa9e39c38a0c990ed67c88342c4df5"}}
result om_d380bfe188f37d82e59eb080_d 481 bytes
call om_d380bfe188f37d82e59eb080_s1 refund_order {"order":"A-1001"}
Plan: clarify
Verdict: passed
Reply: Would you like a refund for this order?
d:refund: Unsure (abstained)
Order A-1001 is delivered
Design: support sha256:bb5f7c6c348ad87ea480b1868416e93624aa9e39c38a0c990ed67c88342c4df5
Record: sha256:c13cfc133298b93bcdcada917caec734bee95f22bb35ac6858c8fc0811c9e2eb
```

## Hand the calls to a Pydantic AI agent

`ommat.conversation.pydantic_ai.to_model_messages` renders the same calls as Pydantic AI
messages. Each pending call names its step's tool and passes the step's arguments by their
names in the design, so the agent's tool takes those names as its parameters; a tool whose
parameters differ fails validation, and Pydantic AI hands the model a retry instead of running
it. The example's tool takes the refund step's `order`:

```python
@agent.tool_plain
def refund_order(order: str) -> dict[str, str]:
    print("refunding", order)
    return {"refund": f"R-{order}"}
```

Run the agent with the history and no new prompt, and it runs the pending calls with its own
tools before it answers:

```python
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(str(STATE["message"]))])]
    history += to_model_messages(transcript(decision.record), model_name="ommat")
    result = asyncio.run(agent.run(None, message_history=history))
```

Run it with `asyncio.run(agent.run(...))` rather than `agent.run_sync`, which leaves an event
loop open. With pydantic-ai-slim 2.49, whose `anthropic` extra sets no upper bound, a fresh
install took anthropic 2.3.0 and its first request failed; pin `anthropic<2` beside
`pydantic-ai-slim[anthropic]` there (1.8.0 worked).

The source calls and decide are ommat's bookkeeping, and the messages that hold them are marked
so in their `metadata`, which Pydantic AI keeps and never sends to a model. The pending calls
are not: the agent ran them. The agent drops an earlier turn's bookkeeping through
`drop_earlier_bookkeeping`, a history processor, while the latest record's stays for the model
to answer from:

```python
agent = Agent(FunctionModel(front), capabilities=[ProcessHistory(drop_earlier_bookkeeping)])
```

A return leaves only with its call, even where Pydantic AI has merged it into the next turn's
request, so the history the model sees never holds one without the other.
`is_bookkeeping(message)` tells a marked message, for code that reads the conversation back,
such as a later turn's state. `ommat.conversation` has the same pair for Chat Completions
messages, which it tells by their ids.

In `agent_run.py`'s second turn the plan clarifies, and the caller appends the reply itself:

```python
    reply = unsure.plan.reply
    if unsure.plan.kind in ("clarify", "inform") and reply is not None:
        history.append(ModelResponse(parts=[TextPart(reply)], model_name="ommat"))
        print(reply)
```

`agent_run.py` prints:

```console
refunding A-1001
Your refund is on its way.
Would you like a refund for this order?
8 bookkeeping messages, 4 kept
```

## Switch parts off

Every part the block includes by default has a switch. `Context(entries, plan=False)` leaves
out the plan's kind, verdict and text, `steps=False` its steps, and `record=False` the design
and record hash; the entries are always given. With no block, `handoff` returns `None` and
nothing is handed off. The switches shape the handoff alone: the transcript gives every call
and the whole conclusion, with or without a block.
