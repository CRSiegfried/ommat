# Hand off a turn

This guide gives the next model in a conversation what a turn settled: the calls the turn made
and the decision it ended in, as tool calls and their results, with the words you choose beside
them. Use it whenever a turn hands on to a generative model, as when a design has done what it
can and an agent finishes the request. The renderers give the turn as an agent's workflow: a
source that called your tool, a read or a write, is a call in the history, so the agent sees
what the turn already did, and you rebuild nothing from the record's `t:` values. It assumes
you have done [Write your first design](../tutorials/write-a-design.md). The spec's
[escalation view](../spec/README.md#the-escalation-view) and
[conversation renderers](../spec/README.md#the-conversation-renderers) state the rules; the
guide shows them at work.

The turn history is not an audit trail, and a renderer says nothing of its own beyond fixed
labels. It is minimal: what the next model needs to carry on, not every value the turn wrote.
Each `t:` value is carried whole, the customer's words and any `identifier` field among them,
so a binding's values set most of the history's size ([Shape a tool's
value](add-a-source.md#shape-a-tools-value)). For the whole turn, read the record, its outline
or the record lens; to share a turn, share its redacted view.

The example's files are in `docs/how-to/hand_off_a_turn/`: `support_design.py` is the design,
`turn_run.py` the driver, which answers through `ommat.bindings.fixed`, so it runs with no
model, and `agent_run.py` a Pydantic AI agent driven by a stand-in function, which acts on the
decision itself. Run each from that folder with `python turn_run.py`, in an environment where
ommat is installed; `agent_run.py` needs `ommat[pydantic-ai]`.

## Give the model the turn's calls

To the next model a source is a tool, so `to_chat_completions(record, loaded)` renders each
source call as an assistant tool call with empty arguments and its `tool` result, then a
`decide` call. A result gives each ref's answer: a question's top label and its probability, a
`t:` value whole, and a marker as its kind and cause code. Decide's arguments are the board
entries the policy reads, and any marker; leave out `loaded` and they are the whole board.
Decide's result is the decision, or for an undecided turn why: its kind, root cause and the
refs it names. In the driver, `STATE` is the turn's state, a dict of the customer's message, email
and order, and `answering(p)` is a `fixed` binding that answers the refund question `yes` with
probability `p`:

```python
    decision = decide_sync(loaded, STATE, {"default": answering(0.9)})
    print(next_model_note(decision))
    for message in to_chat_completions(decision.record, loaded):
        if message["role"] == "assistant":
            for call in message["tool_calls"]:
                print("call", call["id"], call["function"]["name"], call["function"]["arguments"])
        else:
            print("result", message["tool_call_id"], len(message["content"]), "bytes")
```

Append the messages after your own user message for the turn. Each id comes from the record
hash, so one record always renders to the same messages, byte for byte. The same design, state
and answers give the same record: when a turn can repeat one exactly, as a retry does, pass
each rendering its own `turn`, such as the turn's number,
`to_chat_completions(record, loaded, turn=3)`,
so no tool-call id appears twice in the conversation, which a provider may refuse. A source's
tool name is its name with each `/` as `-`; the renderers raise `ValueError` for one a provider
would refuse.

## Word your own note

The engine words nothing. If the next model should also be told what the turn decided, write
that in your code from the `Decision` and the record. `decision.estimate` is the policy's
estimate, and `escalation(record)` routes a turn whose estimate is a marker by its root cause. The
example's policy passes an `Unavailable` on, so a turn whose source could not answer is undecided:

```python
def next_model_note(decision: Decision) -> str:
    """The caller's own words for what the turn settled."""
    escalated = escalation(decision.record)
    if escalated is not None:
        return f"The turn is undecided: {escalated['root_cause']['code']}."
    return f"The turn decided {decision.estimate}."
```

Place the text where your framework takes per-run guidance, such as Pydantic AI's
`instructions` or a system message. A text built from a field the customer wrote carries their
words as written: fence it as data in a user message instead.

The driver prints the first turn's text and calls, then the text of a second turn the policy
leaves to ask the customer and of a third turn whose source could not answer:

```text
The turn decided refund_order.
call om_99095d193e254d10ac37189b_c1 ask {}
result om_99095d193e254d10ac37189b_c1 24 bytes
call om_99095d193e254d10ac37189b_d decide {"d:refund":"yes"}
result om_99095d193e254d10ac37189b_d 27 bytes
The turn decided ask_if_refund.
The turn is undecided: unsupported.
```

The turn ends in decide's result, with no call pending. Here your code
reads the decision and runs its own action when the decision says so, then tells the model
what ran in your own message. When the decision is words for the user, send them and append
them as the assistant's message, so the next turn's history holds what was said; no model call
words it again.

## Hand the calls to a Pydantic AI agent

`ommat.conversation.pydantic_ai.to_model_messages` renders the same calls as Pydantic AI
messages. The messages hold the source calls and decide, and no pending call. The caller acts
on the decision itself, before the agent answers. The example's action takes the order id:

```python
def refund_order(order: str) -> dict[str, str]:
    """The caller's own action on the decision ``refund_order``."""
    print("refunding", order)
    return {"refund": f"R-{order}"}
```

Run the agent with the history and no new prompt, after the caller's action:

```python
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(str(STATE["message"]))])]
    history += to_model_messages(decision.record, loaded, model_name="ommat")
    if decision.estimate == "refund_order":
        refund_order(str(STATE["order"]["id"]))  # type: ignore[index]
    result = asyncio.run(agent.run(None, message_history=history))
```

Run it with `asyncio.run(agent.run(...))` rather than `agent.run_sync`, which leaves an event
loop open. With pydantic-ai-slim 2.49, whose `anthropic` extra sets no upper bound, a fresh
install took anthropic 2.3.0 and its first request failed; pin `anthropic<2` beside
`pydantic-ai-slim[anthropic]` there (1.8.0 worked).

The source calls and decide are ommat's bookkeeping, and the messages that hold them are marked
so in their `metadata`, which Pydantic AI keeps and never sends to a model. The agent drops an
earlier turn's bookkeeping through `drop_earlier_bookkeeping`, a history processor, while the
latest record's stays for the model to answer from:

```python
agent = Agent(FunctionModel(front), capabilities=[ProcessHistory(drop_earlier_bookkeeping)])
```

A return leaves only with its call, even where Pydantic AI has merged it into the next turn's
request, so the history the model sees never holds one without the other.
`is_bookkeeping(message)` tells a marked message, for code that reads the conversation back.
`ommat.conversation` has the same pair for Chat Completions messages, which it tells by their
ids.

In `agent_run.py`'s second turn the decision is `ask_if_refund`, and the caller appends its
question itself:

```python
    if unsure.estimate == "ask_if_refund":
        reply = "Would you like a refund for this order?"
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
