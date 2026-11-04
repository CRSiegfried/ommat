# Decide with real models through idia

In this tutorial you bind an ommat design to two real decision models through the idia
binding, decide two street reports, read what the record holds about each call, replay the
records, and answer a turn again from a stored record. You need Python 3.11 or later, an OpenAI
API key, and a Cloudflare account id and API token with Workers AI access. It builds on
[Write your first design](write-a-design.md), which introduces the answer kinds, markers and
replay.

It uses the street-reports design from the ommat repository:
`docs/tutorials/street_reports/street_reports_design.py` holds the design, and
`docs/tutorials/decide_with_idia/decide_live.py` runs it. Clone the repository to have both.

## Install

idia and ommat's binding to it are installed from their git tags until both are on a
package index. idia's `openai` and `cloudflare` extras bring the clients the two models
need:

```sh
pip install "idia[openai,cloudflare] @ git+https://github.com/CRSiegfried/idia@v0.1.0"
pip install "ommat[idia] @ git+https://github.com/CRSiegfried/ommat@v0.4.0"
```

Each model reads its credentials from the environment. Set `OPENAI_API_KEY` for OpenAI's
Decisions, and `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` for Cloudflare's Clairic.

## The design's three oracles

A source names the oracle that answers it, and `decide` takes one binding per oracle name.

1. Open `street_reports_design.py`. The `triage` source asks a `noul`, a `choice` and a `score`
   and names `oracle="classifier"`. The `details` and `timing` sources ask extractions and
   both name `oracle="extractor"` and are ready in the same round, so one call answers them.
   The `registry` source provides a `t:` ref, a value your own code fetches, under
   `oracle="registry"`.

2. Each model oracle gets one model: OpenAI's Decisions answers label kinds and Cloudflare's
   Clairic answers extractions. The registry is a plain function in `decide_live.py`, which
   says Station Road already has an open work order.

## Bind each oracle

1. Read `live_providers` in `decide_live.py`. `idia.provider(name)` builds a provider, an
   idia `Dispatcher` over one model, which the `async with` block closes when it ends:

   ```python
   async with (
       idia.provider("openai-decisions") as classifier,
       idia.provider("cloudflare-clairic") as extractor,
   ):
       yield classifier, extractor
   ```

2. Read `run`. `oracle_binding(provider)` makes each provider a binding, and the registry's
   function binds as it is:

   ```python
   bindings = {
       "classifier": oracle_binding(classifier),
       "extractor": oracle_binding(extractor),
       "registry": registry,
   }
   ```

   `oracle_binding` takes a `Dispatcher` over one model with its re-ask off, its overflow
   policy `Fail` and its probability decimals within 0 to 17, as `idia.provider` builds
   these two, and refuses any other with `ValueError`. It follows idia while idia is 0.x: it
   may change in any minor release when idia changes, and the changelog will say so.

3. Each call to a binding receives `seconds`, the time the call has: the batch's timeout, 60
   seconds here since no source declares one, or the time left to the turn's deadline when
   one is set and that is smaller. The binding makes it the deadline of every request and
   retry the call sends, and ommat's own timeout ends the call at that same instant.

## Decide two reports

1. `run` decides each of the two reports with `await decide(...)`, prints the decision and each
   call, and replays the record. Run it from its folder:

   ```
   cd docs/tutorials/decide_with_idia
   PYTHONPATH=../street_reports python decide_live.py
   ```

   This is what one run printed:

   ```
   decision: Unsure (abstained on q:triage/hazard)
     call 3: ['details', 'timing'], precision unrounded, attempts 2
       details: no-snapshot snapshot: the wire reported the short name 'clairic', with no version: alias drift cannot be detected from the response
       timing: no-snapshot snapshot: the wire reported the short name 'clairic', with no version: alias drift cannot be detected from the response
     call 4: ['registry'], precision None, attempts None
     call 5: ['triage'], precision 2, attempts 1
       triage: no-snapshot snapshot: the wire reported 'gpt-6-luna', an alias with no snapshot; drift behind it cannot be detected from the response
       triage: unchecked-limit max_input_tokens: max_input_tokens is unknown, so it was not checked
       triage: unchecked-limit max_question_tokens: max_question_tokens is unknown, so it was not checked
       triage: unchecked-limit max_body_bytes: max_body_bytes is unknown, so it was not checked
       triage: unchecked-limit max_levels: max_levels is unknown, so it was not checked
       triage: unchecked-limit min_levels: min_levels is unknown, so it was not checked
       triage: unchecked-limit max_total_options: max_total_options is unknown, so it was not checked
     replay findings: []
   decision: send_crew
     call 3: ['details', 'timing'], precision unrounded, attempts 2
       details: no-snapshot snapshot: the wire reported the short name 'clairic', with no version: alias drift cannot be detected from the response
       timing: no-snapshot snapshot: the wire reported the short name 'clairic', with no version: alias drift cannot be detected from the response
     call 4: ['registry'], precision None, attempts None
     call 5: ['triage'], precision 2, attempts 1
       triage: no-snapshot snapshot: the wire reported 'gpt-6-luna', an alias with no snapshot; drift behind it cannot be detected from the response
       triage: unchecked-limit max_input_tokens: max_input_tokens is unknown, so it was not checked
       triage: unchecked-limit max_question_tokens: max_question_tokens is unknown, so it was not checked
       triage: unchecked-limit max_body_bytes: max_body_bytes is unknown, so it was not checked
       triage: unchecked-limit max_levels: max_levels is unknown, so it was not checked
       triage: unchecked-limit min_levels: min_levels is unknown, so it was not checked
       triage: unchecked-limit max_total_options: max_total_options is unknown, so it was not checked
     replay findings: []
   decision: Unsure (abstained on q:triage/hazard)
     call 3: ['details', 'timing'], precision None, attempts None
       answered from record sha256:1d5b648c8570d81e2a36860489e0dc00dfee77004d1989cb6a5b5fc47982aada
     call 4: ['registry'], precision None, attempts None
       registry: unsupported: the idia binding asks q: refs only, not t:open_orders
     call 5: ['triage'], precision None, attempts None
       answered from record sha256:1d5b648c8570d81e2a36860489e0dc00dfee77004d1989cb6a5b5fc47982aada
   ```

2. The first report is undecided: the classifier put 0.58 on a danger to people, below the
   design's bar of 0.6, so the rule abstained and the policy's estimate is that `Unsure`,
   naming the answer that fell short. The second is decided, `send_crew`: the classifier
   judged a deep pothole a danger.

## Read each call

Each line under a decision is one call item of the record: the sources it asked, its `precision`,
and from the record's `unhashed` part its `attempts` and its `warnings` lines.

1. `precision` is how the model's provider declares it rounds probabilities: 2 decimals for
   Decisions, and `unrounded` for Clairic, which rounds none. The registry call asked no model,
   so it has no precision and no attempts.

2. `attempts` counts the requests the binding sent for the call, retries included: one for
   `triage`, and one each for `details` and `timing`, so 2 on their call.

3. The `warnings` lines are idia's notes on each request, prefixed with the source they
   belong to: `no-snapshot` says the model reported no version, so a change behind its name
   cannot be seen, and `unchecked-limit` names a limit Decisions does not publish. The binding
   writes its own lines there too.

4. The call item also states who answered and how, the provider, the models, the revision,
   the runtime and the elicitation, and its `calls` entry gives a reason for each that is
   null, such as `not-pinned` for Clairic's short name;
   [The provenance profile](../spec/README.md#the-provenance-profile) lists them.

5. A ref the binding cannot answer is a marker, never left out, and the binding's reason, when
   it gives one, is a `warnings` line of that call. A ref whose request ran out of the call's
   time is `timed-out`, and one whose request failed is `raised`; the call's other refs stand.
   The last block below shows an `unsupported` ref; the spec's
   [idia binding](../spec/README.md#the-idia-binding) section lists every marker the
   binding writes and why.

## Replay

1. Each turn printed `replay findings: []`. `run` reads each record back from its stored
   bytes with `read_record` and replays it with `await replay(record, loaded)`: replay answers
   every source from the record and calls no binding, neither a model nor the registry, and an
   empty list says every entry and the decision came out as recorded.

## Answer a turn from a stored record

1. The last block of the output is the first report decided again with every model answer
   from its stored record: `from_record(record, loaded)` is bound under `classifier` and
   `extractor`, and no model is called.

   ```python
   answered = {
       "classifier": from_record(first, loaded),
       "extractor": from_record(first, loaded),
       "registry": oracle_binding(classifier),
   }
   ```

2. It gives the first turn's decision again: the policy's estimate is the `Unsure` that
   `d:hazard` abstained with, as in the first turn.
   Each call it answered opens its `warnings` with `answered from record` and the stored
   record's hash, and holds no precision and no attempts, since it called no model.

3. The registry is bound to the idia binding here, which asks models questions and does
   not fetch a `t:` value. So `t:open_orders` is `unsupported`, with the reason
   `registry: unsupported: the idia binding asks q: refs only, not t:open_orders`. A tool
   value is your code's to fetch, as the registry function does.
