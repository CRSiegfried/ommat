# Changelog

All notable changes to ommat are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor release may break the API; each break is listed under **Changed** or **Removed**.

## [Unreleased]

## [0.4.0] - 2026-10-23

### Added

- A tutorial, `docs/tutorials/decide-with-idia.md`: the street-reports design decided with OpenAI's Decisions and Cloudflare's Clairic through `ommat[idia]`, each call's record read, replayed, and answered again from a stored record. Its script is in `docs/tutorials/decide_with_idia/`, and its test replays the live run from recorded logs.
- `ommat[idia]` and `ommat.bindings.idia`: `oracle_binding(provider)` binds an oracle to a provider through an idia `Dispatcher` over one leaf, one idia request per source under the call's `seconds`, each answer, refusal, fault and failed request mapped to the value or marker ommat takes. Install it with `pip install "ommat[idia] @ git+https://github.com/CRSiegfried/ommat@v0.4.0"`, after idia 0.1.0 with the extras for its providers, as the README shows.
- `seconds`: interpret calls each binding as `binding(batch, seconds=s)`, `s` the time the call has, the smaller of the batch's timeout and the time left to the deadline. A binding converts it to its provider's deadline. A binding that does not take the keyword raises `TypeError` at its first call, so the call is `Failed` with `raised` on every ref of the batch.
- A binding may mark a ref `Unavailable` with `timed-out` or `raised` when the provider request that carried it ran out of time or failed, with the request's status and message as `warnings` lines; the other refs of the call stand, and the call is `returned`. Under a `returned` call these markers are the binding's; on a `Failed` call, the engine's. Replay returns them as recorded.
- The image need: a need whose value is exactly `{"media_type": ..., "data": ...}`, `media_type` naming an image media type such as `image/png` and `data` the image's bytes, declared `record(media_type=string, data=bytes_)`, is an image; a binding whose provider takes images sends it as one, and `ommat.bindings.image_need` reads it.
- `ommat.bindings`, the binding kit, what a binding author calls, which `import ommat` does not import, its seven names public under the same deprecation rule as `ommat.__all__`: `RequestProvenance`, one request's provenance, which `call_result` folds; `extraction_value(value, guess, p_null, confidence, span)` returns the five-member dict an extraction question's ref takes, with each `date`, `time`, `datetime` and `timedelta` written in ommat's grammar, and raises `ValueError` for a value the grammar cannot hold; `bounded_warnings(lines)` returns a `warnings` tuple that conforms, escaping surrogates, leaving out blank lines, and past either bound ending with a line counting those left out; `json_value(value)` returns a need's value in record form, each answer as its document and a tuple as a list, and raises `TypeError` for `bytes` at any depth; `image_need(value)` returns an image need's `(media_type, data)`, or `None`; `prompt_text(prompt)` returns a mapping prompt as `json.dumps(prompt, sort_keys=True)`, nested mappings as objects, and a `str` or `None` as given; `call_result(values, requests, *, precision, warnings=())` folds each request's `RequestProvenance` into the call's `CallResult`, its distinct `snapshot`, `elicitation` and `prompt` one per line, `usage` and `attempts` summed, `truncated` sorted, and `attempts` and `precision` `None` when no request made an attempt. The idia binding uses all seven.
- `ommat.bindings.conformance.check(binding, *, batches=None, deterministic=False)`: the binding conformance suite. It calls a binding with the given batches, or with its own covering each answer kind, an image need and a `t:` source, and returns the protocol's failures as strings naming their check: `seconds`, `returns`, `values`, `markers`, `answers`, `per-ref-cause`, `warnings`, `precision`, `unsupported-provenance` and `deterministic`; an empty list passes.

### Changed

- jurat is renamed Ommat: the distribution, the package and the import are `ommat`, the extra `ommat[idia]`, and Idia is apostille renamed.
- Records and design documents are format 4. Every design hash and record hash changes, so golden hashes must be updated. `read_record` reads formats 1 to 4 and `load` loads documents of formats 1 to 4 to their own hash; a record before format 4 holding a binding's `timed-out` or `raised` under a `returned` call is refused.
- A binding is `async def binding(batch, *, seconds)`: every binding must take the keyword. The fixture binding, `from_record` and the tutorial's binding take it and ignore it.
- The tutorial's binding, `fixed_answers`, reads a `score` by its kind and marks a `t:` ref it does not answer `unsupported`, so it passes the conformance suite's own batches; the tutorial says what `seconds` is for.
- `precision` is the provider's declaration, recorded, and the engine does not check probabilities against it; a `noul`'s `no` is held as the binding's provider library derives it, neither rounded.
- `RenderedQuestion` and `SourceCall` check their fields when built, raising `TypeError` for a field of the wrong type and `ValueError` for a wrong value. `kind` and `value_type` take a member or its string and hold the member; a `noul` question's labels are `yes` and `no` and an extraction question's none; `descriptions` is as long as `labels`; an extraction question needs its `value_type` and `text`, and a question of another kind leaves its extraction members unset; a source call's `refs` and `questions` are not both non-empty. A string subclass is held as its string data. `CallResult` still checks nothing.
- `from_record` bounds its `warnings` with `bounded_warnings`, and refuses with `unsupported` a source whose need is not a ref or holds a value outside the value domain, where it raised.
- Load check 4 refuses a source that provides no ref: a source provides one or more `q:` or `t:` refs. A design with such a source loaded before and now raises `InvalidDesign`.
- The redacted view digests a requirement's value, on a plan's step and in a refused verdict's failure, under its ref's class tree, as that ref's entry is; the value was open.

### Fixed

- A `-0.0` a binding, a rule or the policy returns is held as `0.0` on the board, in an answer, in the plan and in the record's stored copy, where it stayed `-0.0` live while the record wrote `0` and replay read back `0.0`, so a rule that told them apart diverged in replay.
- The design lints probe both integers either side of a boundary that is not integral on an integer read, where they dropped it and could miss the values below or above it, so `dead_questions` could call a live question `no-change-found`.
- `t.list_`, `t.mapping` and `t.record` raise `ValueError` for a type nesting past the depth bound, where they returned a string load refused.
- `run_plan` runs a decision whose record holds a marker entry, where it raised `KeyError` before the first tool call.
- A marker a rule returns is `invalid-return` when a ref of its cause is not a ref the design provides, a source, rule or tool name, a parameter path, `policy`, `codec`, `rule:N` or `step:N`, where it was written as given: free text in a cause reached the record's hashed log and stayed open in the redacted view.
- `load` gives a function whose `__wrapped__` chain loops a null source hash, as for any function whose source cannot be read, where it raised `ValueError`.

## [0.3.0] - 2026-10-22

### Added

- A tutorial, `docs/tutorials/write-a-design.md`: one offline design taken through `load`, `decide`, the record and replay, its example tested. Its example files are in the repository at `docs/tutorials/write_a_design/`, for a reader who installed jurat from an index.
- Label descriptions: a `choice` or `score` label may be `{"name": ..., "description": ...}`, which the design document, the call item and the question's identity hold as declared. `RenderedQuestion.descriptions`, keyword-only, gives each label's description aligned with `labels`, `None` where a label has none. Answers, the answer grid, the lints and the `boundaries` checks read names only.
- `CallResult.warnings`: the binding's own lines about the call, recorded under the call in `unhashed` and outside every hash. A malformed value is kept in `invalid` and is never a fault.
- `CallResult.precision`: the decimal places the provider rounds probabilities to, hashed on the call item. A value that is not `None` or an integer from 0 to 17 is `invalid-return` for the whole call.
- `CauseCode.unsupported`: the marker a binding returns for a ref it will not put to any provider, with the reason as a line of `warnings`. It is the 24th cause code. A rule that returns the marker it read provides it unchanged, so `unsupported` stands on a `d:` entry with its refs as any binding code does.
- The record document's `hash` member: the record hash over the log. `read_record` recomputes it and refuses a document whose `hash` is missing or does not match.
- `jurat.evaluation`, the evaluation subpackage, with its first job: `use_answers(record, loaded, other)` and `use_answers_sync`, which rerun `record`'s turn under `loaded` with `other`'s calls answering and return `{"other", "findings"}`, `other` its record hash and the findings replay's. Both records are a `Record` or a stored document and must be finished turns of the loaded design with the same state entry and `settings` item. Each refusal is an `EvaluationRefusal`, its `reason` a `RefusalReason`; a replay record is a valid `record` or `other`, so two models decided under different parameter sets compare by replaying the second under the first's loaded design.
- `jurat.from_record(record, loaded)`: an oracle binding that answers a live turn's sources from a stored record, refusing with `unsupported` a source with no returned call or different needs, a changed question, a ref with no stored entry and a recorded marker other than `invalid-answer` or `invalid-return`, with null provenance and a `warnings` line naming the record's hash.
- `ReadView`, the view a function `when`, rule or policy is called with, is public, so a component can be annotated with it. `When`, `Rule` and `Policy` type `function` as `Callable[[ReadView], object]`, so a lambda passes pyright strict.

### Changed

- Records and design documents are format 3. Every design hash and record hash changes, so golden hashes must be updated. Every format-3 call item holds `precision`.
- An exception's recorded traceback starts at the component's own frame and names each file by its module, so the record holds no local paths from the traceback.
- `read_record` reads format-1 and format-2 records and `load` loads format-1 and format-2 design documents to their own hash, each under its format's rules. A format-2 record holding a described label, `precision`, `warnings`, `unsupported` or a `hash` member is refused.
- `Step` takes its arguments as any `Mapping`, read once through `items()`, and still keeps a read-only copy; a non-mapping is a `TypeError`. The recorded plan is unchanged.
- Check 4's message for a rule that provides another ref names the rule's own `d:` ref, and for another rule's `d:` ref, whose it would be.
- A `load` refusal for a type that does not fit names the type as written: check 9's text and `reference_date` refs, check 10's state declaration and parameter shape, check 8's boundary leaf, and check 9's boundary key and role leaves.

## [0.2.0] - 2026-10-22

### Added

- The `extraction` answer kind: a question that asks for a typed value read from a text. `Question` takes `value_type`, `is_list`, `text` and `reference_date`; `ValueType` names the six value types; `ExtractionAnswer` holds the answer's `value`, `guess`, `p_null`, `confidence` and `span`, and checks each when built. Dates, times, datetimes and durations are checked by jurat's own grammar, alike on every Python.
- `RenderedQuestion` gives an extraction question's `value_type`, `is_list`, `text` and `reference_date`, keyword-only; a question of another kind leaves them at `None`, `False`, `None` and `None`.
- `CallResult.truncated`: the `q:` refs whose answer came from a response the provider cut short, recorded under the call in `unhashed` and outside every hash.
- `jurat.helpers.extract`: an extraction answer's value, or `Unsure` with `abstained` when the value is null or `p_null` is at or above the bar.
- The extraction roles of a `boundaries` entry, `null`, `confidence` and `value`, and the dead-question lint's points over an extraction read; an extraction read with no `value` role takes its values from the lint's `values` argument.

### Changed

- Records and design documents are format 2, and the helper version is 2. Every design hash and record hash changes, so golden hashes must be updated.
- `AnswerKind` has a fourth member, `extraction`; code that matches on every kind must handle it.
- The ledger refuses a `q:` answer whose kind is not its call item's question's.
- 0.2.0 reads format-1 records and loads format-1 design documents to their own hash, each under format 1's rules, so a record written under 0.1.0 replays under its stored design with `helpers-mismatch` as its only finding.

## [0.1.0] - 2026-10-21

### Changed

- `Source.batch` is now `Source.oracle`, the name of the binding that answers the source, and the design document's `batch` member is `oracle`. Every design hash changes, so a record written under 0.0.1 names a design hash that no reloaded design has. `EntryRefusal` messages name the oracle, and a missing `default` binding says it serves the sources with no oracle.
- `load` refuses a source that provides both `q:` and `t:` refs (check 4). Split it into two sources that name one oracle: ready in the same round, they are still one call.

## [0.0.1] - 2026-10-20

### Added

- The design types: `Design`, `Source`, `Question`, `Text`, `Rule`, `Policy`, `When`, `Requirement`, `SourceRequirement`, `Group` and `Parameter`, with `Table`, `PolicyTable` and `PlanDeclaration` for the forms `load` refuses in this release.
- `jurat.types`: constructors for the type a field holds (`boolean`, `integer`, `number`, `string`, `bytes_`, `label`, `list_`, `mapping`, `record` and `nullable`).
- `load`: checks a `Design`, or its canonical document, before any turn runs and returns a `LoadedDesign`; an invalid design raises `InvalidDesign` naming each failed check.
- `decide` and `decide_sync`: one turn, interpret over rounds of sources and rules, then compile with the policy and the guard, returning the plan, the verdict and the record. Rules and the policy are Python functions.
- `RunSettings`: the run settings, with a round bound of 8 and a 60 s timeout by default; `EntryRefusal` for a wrong argument at `decide`'s entry, raised before any record is written.
- The markers `Unsure`, `Unavailable` and `Unasked`, each with its cause, for every value a turn could not settle.
- The record: its canonical JSON encoding and hashes, the observer, and `read_record` for a stored record, which raises `InvalidRecord` for one it cannot read.
- `replay` and `replay_sync`: a recorded turn rerun under a loaded design with no model call, reporting each difference as a finding.
- The record views: `escalation` and `redact`.
- The binding protocol between the engine and a model adapter: `SourceCall`, `RenderedQuestion` and `CallResult`.
- `jurat.helpers`: the choice helpers and combinators rules call, the design lints `dead_questions` and `unsure_to_execute`, and the reference runner `run_plan`, which raises `RunRefusal` for a plan it refuses before calling any tool and returns `BindingNotFound` for a step argument whose binding missed.
- Project scaffolding: package, license, code of conduct, contributing and security policies, CI.

### Not included

- `load` refuses decision tables, policy tables, data rules, plan declarations, candidate sets, candidate-valued label sets and grounded declarations.
- No state carries from one turn to the next.

[Unreleased]: https://github.com/CRSiegfried/ommat/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/CRSiegfried/ommat/releases/tag/v0.4.0
[0.3.0]: https://github.com/CRSiegfried/jurat/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CRSiegfried/jurat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CRSiegfried/jurat/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/CRSiegfried/jurat/releases/tag/v0.0.1
