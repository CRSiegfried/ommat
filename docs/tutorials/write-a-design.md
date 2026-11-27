# Write your first design

In this tutorial you build an ommat design for a shop's help desk, decide two customer
messages offline, and replay the records. You need Python 3.11 or later and ommat 0.4.0,
installed as the [README](../../README.md#install) shows.

## The design you'll build, and where it lives

The help desk asks what a message is about, whether it is urgent and which order number it
quotes, then routes or escalates it. The finished files are in
`docs/tutorials/write_a_design/`: `helpdesk_design.py` holds the design and
`helpdesk_run.py` runs it. If you installed ommat from a package index, fetch the two
files from the repository at
[`docs/tutorials/write_a_design/`](https://github.com/CRSiegfried/ommat/tree/main/docs/tutorials/write_a_design).

Keep the design in its own module and import it from your script, as this example does. A
component's module is part of the design's hash, so the same design run as the launched
script, `__main__`, gets a different hash and its records replay with `design-mismatch`
(the spec's [Replay](../spec/README.md#replay) section says how to find the cause).

## Sources and oracles

A source fetches answers from outside. It declares the refs it provides, the board values
it needs, and the oracle that answers it.

1. Open `helpdesk_design.py` and find the first source, `triage`, which asks two questions:

   ```python
   SOURCES = {
       "triage": Source(
           {
               "q:triage/urgent": Question("noul", Text("Does the customer need an answer today?")),
               "q:triage/topic": Question(
                   "choice", Text("What is the customer's message about?"), TOPICS
               ),
           },
           needs=["t:state"],
           oracle="classifier",
       ),
   ```

   The key, `triage`, is the source's id. Sources and rules share one set of names.

2. `needs=["t:state"]` hands the source the turn's state. `oracle="classifier"` names the
   binding that answers it: at decide you give one binding per oracle name. A source with no
   `oracle` is answered by the binding named `default`, and all such sources ready in a round
   go to it in one call.

3. Find the second source, `lookup`, with the same `needs` and `oracle="reader"`. Its one
   question, `q:order/number`, comes in the next section.

4. At the bottom, the design's `state=t.record(message=t.string)` makes each turn's state
   a record of one string, `message`.

## Questions

Each `q:` ref maps to a `Question`: its kind, its wording as a `Text`, and any labels.

1. The urgency question is a `noul`, a yes-or-no question. It takes no labels.

2. The topic question is a `choice`. Its labels can be plain names, such as
   `["delivery", "return"]`, but a name alone leaves the model to guess what it covers. A
   described label is exactly `{"name": ..., "description": ...}`:

   ```python
   TOPICS = [
       {"name": "delivery", "description": "Where an order is, or when it will arrive"},
       {"name": "return", "description": "Sending something back, or a refund for it"},
       ...,  # "damaged" and "other", described the same way
   ]
   ```

   Answers and rules use the names alone. The description travels with the question to the
   binding, and changing it makes a different question.

3. Find `lookup`'s extraction question, which asks for a typed value read from a text:

   ```python
               "q:order/number": Question(
                   "extraction",
                   Text("The order number the customer quotes"),
                   None,
                   value_type="text",
                   text={"ref": "t:state.message"},
               ),
   ```

   Its labels are `None`; [the spec](../spec/README.md#the-extraction-kind) lists the
   value types. `text` is an origin, `{"ref": r}`, naming the value the model reads.

4. The `text` ref must name a value whose type is exactly `string`. `t:state.message` is
   one, so the design loads; declare `message` as `t.integer()` and `load` refuses it.

5. The `text` ref joins the source's needs: `lookup` declared only `t:state`, but its
   binding also receives `t:state.message`.

6. Say what each question is for. A target is the property of the state the design is trying
   to determine. Its statement is a noun phrase saying what the property is, plus at most one
   clause naming the neighbouring thing it is most easily confused with, so a reader can judge
   how a question's answer bears on it. The cases that count go in the question's wording:

   ```python
   TARGETS = {
       "tg:urgent": Target(
           "The customer's need for an answer today, not their impatience.",
           values="yes or no",
       ),
       "tg:topic": Target(
           "The subject the customer most asks to have dealt with, not every subject the "
           "message mentions.",
           values="one of delivery, return, damaged or other",
       ),
   }
   ```

   Each question names its target and takes the cases into its wording, in place of the
   wording from the first step. `q:triage/urgent` names `target="tg:urgent"` and asks
   `Text("Does the customer need an answer today? A deadline, a missed delivery or a safety
   issue counts; impatience alone does not.")`. `q:triage/topic` names `target="tg:topic"`
   and asks `Text("What is the customer's message about? If it raises several things, answer
   for the one it most asks to have dealt with.")`. The design declares
   `targets=TARGETS`. Targets gate nothing: they tell a reader what
   each question is for.

## Rules

A rule turns answers into an estimate, a `d:` ref other rules and your code read. It is a function of a
read view, `r`.

1. Find the topic rule. Its function sits at module level, so a stored design reloads it by
   import:

   ```python
   def topic_of(r: ReadView) -> object:
       """The topic, when its answer clears the bar and the margin; else abstain."""
       return pick_with_margin(r, "q:triage/topic", r.param("desk/topic_at"), r.param("desk/margin"))
   ```

   ```python
   RULES = {
       "topic": Rule(
           "d:topic",
           t.label("damaged", "delivery", "other", "return"),
           topic_of,
           version="1",
           reads=["q:triage/topic"],
           params=["desk/topic_at", "desk/margin"],
       ),
   ```

   The rule provides `d:topic`, one of the four names, listed in code-point order.

2. Every rule has a `version`, its `reads` and its `params`. `load` refuses a rule with no
   version string; raise it when the function changes. `reads` lists every ref the function
   reads; an undeclared read makes the rule's ref `Unavailable` with the cause
   `undeclared-read`. `params` lists the parameter paths it reads with `r.param`. A read
   returns the board's own value, so a rule must not change it: a list sorted or a mapping
   popped in place changes what every later reader of that ref sees. Build a new value.

3. The bars are parameters, leaves of the design's `parameters`, each a type and a value:

   ```python
   PARAMETERS = {
       "desk/topic_at": Parameter(t.number(0, 1), 0.5),
       "desk/urgent_at": Parameter(t.number(0, 1), 0.6),
       "desk/margin": Parameter(t.number(0, 1), 0.2),
       "desk/no_order_at": Parameter(t.number(0, 1), 0.5),
   }
   ```

   A bar written into the function is one the design document cannot show and the lints
   cannot see; as a parameter it is part of the design's hash, so a record names the bars it
   ran under.

4. `pick_with_margin(r, ref, bar, margin)` returns the top label when its probability is at
   least `bar` and leads the second by at least `margin`. Otherwise it abstains, returning
   `Unsure` with the cause `abstained` naming `ref`. A marker it reads, it returns as is.

5. The urgency rule is built the same way, with `t.label("no", "yes")` and the bar
   `desk/urgent_at`. Its key, `urgent`, is the name of the `d:` ref it provides, as every
   rule's must be.

6. The order rule, `d:order` of type `t.string`, calls
   `extract(r, "q:order/number", r.param("desk/no_order_at"))`. `extract(r, ref, null_bar)`
   returns the answer's `value`. It abstains, again with `Unsure` and `abstained`, when the
   value is `None` or the model's `p_null`, the probability that the text holds none, is at
   or above `null_bar`.

An estimate ommat cannot settle is never `None` or a quiet `no` but a marker: `Unsure`,
`Unavailable` or `Unasked`. Each is an `Undetermined`, so one `isinstance` test catches them
all; `if value:` on a marker raises.

## The rule the caller acts on

The route rule is an ordinary rule whose estimate the caller acts on. It reads the other
estimates and returns the name of what the caller should do.

1. Find it in `helpdesk_design.py`:

   ```python
   def route(r: ReadView) -> object:
       """Escalate what the desk cannot settle or must answer today; inform the rest."""
       topic = r.read("d:topic")
       urgent = r.read("d:urgent")
       order = r.read("d:order")
       if isinstance(topic, Undetermined):
           return topic
       if urgent == "yes":
           return "answer_today"
       if isinstance(order, Undetermined):
           return f"route_{topic}"
       return f"route_{topic}_with_order"
   ```

   Each marker is handled on purpose: an unsettled topic is returned as it is, so the
   route is undecided; an unsettled urgency never equals `"yes"`; and a missing order number
   still routes.

2. The design wraps it as a rule, `d:route` of type `t.string`, with
   `reads=["d:topic", "d:urgent", "d:order"]`, in `RULES`, which `Design` takes as
   `rules`, beside `state` and `sources`.

3. The caller reads the route's entry, `decision.record.entries["d:route"]`, which holds a
   `value` or a `marker`, and acts on it; a source could instead act during
   the turn ([Add a source](../how-to/add-a-source.md#a-source-that-changes-state)). When the entry is a marker, the route is
   undecided, and `escalation(record, "d:route")` routes it by its root cause.

## Bindings and decide

A binding is the async function that answers a batch of sources. This one answers every
question the same way each time, so no model is needed.

1. Open `helpdesk_run.py` and read `fixed_answers`. It takes a tuple of `SourceCall`s and,
   as a keyword, `seconds`: the time the call has, which a binding over a model provider
   turns once into its provider's deadline, by which its requests and retries end. It
   returns a `CallResult` whose `values` map each ref to its value. A `noul`, `choice` or
   `score` answer gives a probability to each label name, such as `{"yes": 0.2, "no": 0.8}`.
   A `t:` ref this binding does not answer is `Unavailable(Cause("unsupported"))`, never
   left out; a binding may give its reason as a line of `warnings`. This design has no `t:`
   ref, so that loop covers what another design would send. An extraction answer has exactly
   five keys:

   ```python
   start = call.needs[question.text].find("A-1042")
   found = start >= 0
   values[ref] = {
       "value": "A-1042" if found else None,
       "guess": None,
       "p_null": 0.02 if found else 0.95,
       "confidence": 0.9,
       "span": [start, start + 6] if found else None,
   }
   ```

   `question.text` is the key in `call.needs` that holds the message, and `span` is a pair
   of offsets into it.

2. A binding receives a source's questions in the code-point order of their refs, so
   `q:triage/topic` comes first; that order is part of what ran.

3. This binding is written out to show what a binding does. For a stand-in that answers each
   ref with one value, or a fact your code already holds, `ommat.bindings.fixed({ref: value})`
   builds the binding for you.

4. Read `main`. It loads the design, binds both oracle names, and decides each message:

   ```python
   loaded = load(DESIGN)
   bindings = {"classifier": fixed_answers, "reader": fixed_answers}
   decision = decide_sync(loaded, {"message": message}, bindings)
   ```

   `load` raises `InvalidDesign` naming every failure. Without `"reader"`, `decide_sync`
   raises `EntryRefusal`. A real binding, such as one over a model provider's API, plugs
   in at the same place, in this mapping. So does `ommat.from_record(record, loaded)`,
   which answers an oracle's sources from a stored record while the others run live;
   `record` is a `Record` or a stored document, bytes or parsed, read through `read_record`.

5. Run the driver from its directory:

   ```
   cd docs/tutorials/write_a_design
   python helpdesk_run.py
   ```

   Each turn's first line shows the decision:

   ```
   decision: route_delivery_with_order
   decision: route_delivery
   ```

   In the second, `extract` abstained, so `d:order` is `Unsure`.

## Reading the record

`decision.record.document` is the record as canonical JSON bytes; store them and read them
back with `read_record`. Its `log` is the hashed turn, its items; `unhashed` sits
outside every hash.

1. Find the call items in the log. A call item holds `item`, `round`, `sources`,
   `questions` and `outcome`, and the call's provenance: `provider`, `model_requested`,
   `model_reported`, `revision`, `runtime`, `elicitation`, `precision`, `prompt` and
   `claimed`, all hashed. A binding that sends no request, as this one, leaves them null and
   `claimed` empty.

2. `start`, `latency`, `usage`, `attempts`, `warnings`, `truncated`, `sums`, `unexpected`
   and `invalid` sit under `unhashed["calls"]`, keyed by the call item's log index as a
   string, with `request_id`, `missing`, the reason for each provenance detail
   a binding that sent a request leaves null, and `clients`, each present only when the binding
   gives it.

3. Name each call's oracle. A call item names no oracle, so look up each of its
   `sources` in the design: the source's `oracle`, or `default` when it declares none. The
   provider shows in `provider`, null here. The driver does it:

   ```python
   calls = record.unhashed["calls"]
   for index, item in enumerate(record.log):
       if item["item"] == "call":
           oracles = [DESIGN.sources[s].oracle or "default" for s in item["sources"]]
           latency = calls[str(index)]["latency"]
           print(f"  call {index}: {item['sources']} via {oracles}, {latency:.6f} s")
   ```

   It prints `call 3: ['lookup'] via ['reader']` and `call 4: ['triage'] via
   ['classifier']`, each with its latency.

## Replay

1. Replay each record, as `main` does: `rerun, findings = replay_sync(record, loaded)`.
   Replay answers from the record and calls nothing. Each finding is one difference.

2. Check the findings, and the decision, not `record_hash`, since replay answers
   with null provenance and a real provider's record hashes differently from its replay.
   This offline binding returns null provenance too, so here the hashes happen to match:
   the driver prints `same hash: True`.

3. Each turn prints `replay findings: []`: the same answers rebuilt the same decision.

To compare two models' records of one turn, `ommat.evaluation.use_answers(record, loaded,
other)` reruns `record` with `other`'s answers. It is async, and `use_answers_sync` is its
twin. Each record is a `Record` or a stored document. It returns
`{"other": <other's record hash>, "findings": [...]}`, the findings being replay's.
