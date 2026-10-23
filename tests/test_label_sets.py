# pyright: reportPrivateUsage=false
"""A candidate-valued label set: a ``choice`` whose labels are a fixed part and the members of a
``list<string>`` candidate set, rendered each time the question is asked (§candidate-sets)."""

import json
from typing import Any

import pytest

from ledger_logs import rehashed
from ommat import (
    CallResult,
    Cause,
    CauseCode,
    Design,
    Plan,
    Policy,
    Question,
    Record,
    Rule,
    RunSettings,
    Source,
    SourceCall,
    Text,
    Unavailable,
    Unsure,
    _form,
    decide_sync,
    from_record,
    load,
    read_record,
    replay_sync,
)
from ommat import types as t

SET = "t:sender_candidates"
FIXED = ["none", "several"]


def inform(r: Any) -> Plan:
    return Plan("inform", reply="ok")


def asks(
    labels: Any = None, *, needs: tuple[str, ...] = (), other: bool = True, cost: float = 0
) -> Source:
    provides: dict[str, Any] = {
        "q:sender": Question(
            "choice",
            Text("Who sent it?"),
            {"fixed": FIXED, "candidates": SET} if labels is None else labels,
        )
    }
    if other:
        provides["q:other"] = Question("noul", Text("Is it about something else?"))
    return Source(provides, needs=list(needs), cost=cost)


def design(pick: Source | None = None, **sources: Source) -> Design:
    finder = Source(
        {SET: {"type": t.list_(t.string), "candidates": True}}, needs=["t:state"], oracle="tools"
    )
    return Design(
        name="label-sets",
        state=t.record(candidates=t.list_(t.string)),
        sources={"finder": finder, "pick": asks() if pick is None else pick, **sources},
        policy=Policy(inform, version="1"),
    )


class Model:
    """Answers each question over the labels it was passed: ``top`` at 0.9 when given, else the
    first label; records every batch."""

    def __init__(self, top: str | None = None, answers: dict[str, Any] | None = None) -> None:
        self.top = top
        self.answers = answers or {}
        self.batches: list[tuple[SourceCall, ...]] = []

    async def __call__(self, batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        self.batches.append(batch)
        values: dict[str, Any] = {}
        for call in batch:
            for ref, q in call.questions.items():
                if ref in self.answers:
                    values[ref] = self.answers[ref]
                    continue
                labels = list(q.labels)
                top = self.top if self.top in labels else labels[0]
                rest = 0.1 / (len(labels) - 1) if len(labels) > 1 else 0.0
                values[ref] = {label: 0.9 if label == top else rest for label in labels}
                if len(labels) == 1:
                    values[ref] = {top: 1.0}
        return CallResult(values)


async def finder(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    return CallResult({SET: list(batch[0].needs["t:state"]["candidates"])})


def run(
    candidates: list[str] | None,
    model: Any = None,
    d: Design | None = None,
    tools: Any = finder,
    settings: RunSettings | None = None,
) -> Record:
    state = {"candidates": [] if candidates is None else candidates}
    d = d or design()
    bindings: dict[str, Any] = {"default": model or Model()}
    if "finder" in d.sources:
        bindings["tools"] = tools
    return decide_sync(load(d), state, bindings, settings).record


def entries(record: Record) -> dict[str, dict[str, Any]]:
    return {i["ref"]: i for i in record.log if i["item"] == "entry"}


def marker(record: Record, ref: str) -> tuple[str, str, list[str]]:
    m = entries(record)[ref]["marker"]
    return m["kind"], m["cause"]["code"], m["cause"]["refs"]


def calls(record: Record) -> list[dict[str, Any]]:
    return [i for i in record.log if i["item"] == "call"]


# ---------------------------------------------------------------------------------------------
# Rendering


def test_the_labels_are_the_fixed_part_then_the_candidates_in_the_sets_order() -> None:
    model = Model(top="bob@y.org")
    record = run(["bob@y.org", "ana@x.org"], model)
    (asked,) = [c for b in model.batches for c in b if c.id == "pick"]
    q = asked.questions["q:sender"]
    assert q.labels == ("none", "several", "bob@y.org", "ana@x.org")
    assert q.descriptions == (None, None, None, None)
    assert SET not in asked.needs
    answer = entries(record)["q:sender"]["value"]
    assert answer["labels"] == ["none", "several", "bob@y.org", "ana@x.org"]
    assert answer["probabilities"]["bob@y.org"] == 0.9


def test_a_described_fixed_part_keeps_its_descriptions_and_a_candidate_has_none() -> None:
    labels = {"fixed": [{"name": "none", "description": "No sender is named"}], "candidates": SET}
    model = Model()
    run(["ana@x.org"], model, design(asks(labels)))
    (asked,) = [c for b in model.batches for c in b if c.id == "pick"]
    q = asked.questions["q:sender"]
    assert q.labels == ("none", "ana@x.org")
    assert q.descriptions == ("No sender is named", None)


def test_the_call_item_holds_the_labels_as_declared_and_one_identity_for_two_sets() -> None:
    first, second = run(["ana@x.org"]), run(["bob@y.org", "cy@z.org"])
    (one,) = [c for c in calls(first) if c["sources"] == ["pick"]]
    (two,) = [c for c in calls(second) if c["sources"] == ["pick"]]
    assert one["questions"]["q:sender"]["labels"] == {"fixed": FIXED, "candidates": SET}
    assert one["questions"]["q:sender"]["identity"] == two["questions"]["q:sender"]["identity"]


def test_a_one_label_rendering_is_asked() -> None:
    labels: dict[str, Any] = {"fixed": [], "candidates": SET}
    record = run(["ana@x.org"], Model(), design(asks(labels)))
    assert entries(record)["q:sender"]["value"]["labels"] == ["ana@x.org"]


# ---------------------------------------------------------------------------------------------
# A question left out of its batch


def test_an_empty_set_leaves_the_question_unasked_when_false_and_asks_the_rest() -> None:
    record = run([])
    assert marker(record, "q:sender") == ("Unasked", "when-false", [SET])
    assert entries(record)["q:sender"]["call"] is None
    assert "value" in entries(record)["q:other"]


def test_a_candidate_equal_to_a_fixed_label_is_a_label_collision() -> None:
    record = run(["ana@x.org", "none"])
    assert marker(record, "q:sender") == ("Unavailable", "label-collision", [SET])


def test_a_marker_on_the_set_marks_the_question_alone() -> None:
    async def failing(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        return CallResult({SET: Unavailable(Cause(CauseCode.refused))})

    record = run(["ana@x.org"], tools=failing)
    assert marker(record, "q:sender") == ("Unavailable", "need-unavailable", [SET])
    assert "value" in entries(record)["q:other"]


def unsure_finder(r: Any) -> Any:
    return Unsure(Cause(CauseCode.abstained, ["t:state"]))


def rule_set(function: Any) -> Design:
    labels = {"fixed": FIXED, "candidates": "d:find"}
    base = design(asks(labels))
    return Design(
        name=base.name,
        state=base.state,
        sources={"pick": base.sources["pick"]},
        rules={
            "find": Rule(
                "d:find",
                t.list_(t.string),
                function,
                version="1",
                reads=["t:state"],
                candidates=True,
            )
        },
        policy=base.policy,
    )


def test_an_unsure_set_from_a_rule_leaves_the_question_unasked_need_undetermined() -> None:
    record = run(["ana@x.org"], d=rule_set(unsure_finder))
    assert marker(record, "q:sender") == ("Unasked", "need-undetermined", ["d:find"])


def found(r: Any) -> Any:
    return list(r.read("t:state")["candidates"])


def test_a_rules_set_renders_the_labels() -> None:
    record = run(["ana@x.org", "bob@y.org"], d=rule_set(found))
    labels = entries(record)["q:sender"]["value"]["labels"]
    assert labels == ["none", "several", "ana@x.org", "bob@y.org"]


def test_a_set_holding_the_whole_source_leaves_the_binding_uncalled() -> None:
    model = Model()
    record = run([], model, design(asks(other=False, cost=5)), settings=RunSettings(budget=1))
    assert [c.id for b in model.batches for c in b] == []
    assert [c["sources"] for c in calls(record)] == [["finder"]]
    assert marker(record, "q:sender") == ("Unasked", "when-false", [SET])


def test_a_source_that_declares_the_set_a_need_is_marked_whole() -> None:
    d = design(asks(needs=(SET,)))
    record = run([], d=d)
    # A need holding a value: the empty set is a value, so the source runs and its question
    # alone is unasked.
    assert marker(record, "q:sender") == ("Unasked", "when-false", [SET])

    async def failing(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        return CallResult({SET: Unavailable(Cause(CauseCode.refused))})

    record = run(["ana@x.org"], d=d, tools=failing)
    assert marker(record, "q:other") == ("Unavailable", "need-unavailable", [SET])


def test_a_budget_cut_leaves_a_left_out_question_its_own_marker() -> None:
    record = run(["none"], d=design(asks(cost=5)), settings=RunSettings(budget=1))
    assert marker(record, "q:sender") == ("Unavailable", "label-collision", [SET])
    assert marker(record, "q:other") == ("Unasked", "over-budget", [])


# ---------------------------------------------------------------------------------------------
# The size bound


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_form, "MAX_LABELS", 4)
    monkeypatch.setattr(_form, "MAX_LABEL_CODE_POINTS", 40)


@pytest.mark.usefixtures("small")
def test_labels_at_the_bounds_are_asked_and_beyond_them_overflow() -> None:
    assert "value" in entries(run(["a", "b"]))["q:sender"]
    assert marker(run(["a", "b", "c"]), "q:sender") == ("Unavailable", "label-overflow", [SET])
    # 4 + 7 code points of fixed names, then 29 of candidates: 40 in all.
    assert "value" in entries(run(["x" * 29]))["q:sender"]
    assert marker(run(["x" * 30]), "q:sender") == ("Unavailable", "label-overflow", [SET])


@pytest.mark.usefixtures("small")
def test_the_fixed_parts_descriptions_count_toward_the_code_point_bound() -> None:
    labels = {"fixed": [{"name": "none", "description": "n" * 30}], "candidates": SET}
    d = design(asks(labels))
    assert "value" in entries(run(["abcdef"], d=d))["q:sender"]
    assert marker(run(["abcdefg"], d=d), "q:sender")[1] == "label-overflow"


@pytest.mark.usefixtures("small")
def test_an_overflow_is_named_before_a_collision() -> None:
    record = run(["none", "b", "c"])
    assert marker(record, "q:sender") == ("Unavailable", "label-overflow", [SET])


# ---------------------------------------------------------------------------------------------
# The set's members and the answers


def test_a_sources_set_with_repeated_members_is_invalid_return() -> None:
    record = run(["ana@x.org", "ana@x.org"])
    assert marker(record, SET) == ("Unavailable", "invalid-return", [SET])
    assert marker(record, "q:sender") == ("Unavailable", "need-unavailable", [SET])


def test_composed_and_decomposed_spellings_are_two_candidates() -> None:
    record = run(["é", "é"])
    assert entries(record)["q:sender"]["value"]["labels"][2:] == ["é", "é"]


def repeated(r: Any) -> Any:
    return ["a", "a"]


def test_a_rules_set_with_repeated_members_is_invalid_return_naming_the_rule() -> None:
    record = run(["x"], d=rule_set(repeated))
    assert marker(record, "d:find") == ("Unavailable", "invalid-return", ["find"])
    assert marker(record, "q:sender") == ("Unavailable", "need-unavailable", ["d:find"])


@pytest.mark.parametrize(
    "answer",
    [
        {"none": 0.5, "several": 0.5},  # a candidate left out
        {"none": 0.4, "several": 0.3, "ana@x.org": 0.2, "eve@x.org": 0.1},  # a label not passed
    ],
)
def test_an_answer_not_over_the_rendered_labels_is_invalid(answer: dict[str, float]) -> None:
    record = run(["ana@x.org"], Model(answers={"q:sender": answer}))
    assert marker(record, "q:sender") == ("Unavailable", "invalid-answer", ["q:sender"])


async def extra(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    return CallResult({"q:other": {"yes": 0.5, "no": 0.5}, "q:sender": {"none": 1.0}})


def test_a_returned_key_for_a_left_out_question_is_an_unexpected_ref() -> None:
    record = run([], extra)
    assert marker(record, "q:sender") == ("Unasked", "when-false", [SET])
    assert marker(record, "q:other")[1] == "unexpected-ref"


@pytest.mark.parametrize("code", [CauseCode.label_overflow, CauseCode.label_collision])
def test_a_binding_that_returns_an_engine_label_code_is_invalid_return(code: CauseCode) -> None:
    record = run(["ana@x.org"], Model(answers={"q:sender": Unavailable(Cause(code))}))
    assert marker(record, "q:sender")[1] == "invalid-return"


# ---------------------------------------------------------------------------------------------
# from_record and the fixture binding


def answered_from(record: Record, candidates: list[str], d: Design | None = None) -> Record:
    """A live turn over ``candidates`` whose questions ``from_record`` answers from ``record``."""
    d = d or design()
    loaded = load(d)
    bindings: dict[str, Any] = {"default": from_record(record, loaded)}
    if "finder" in d.sources:
        bindings["tools"] = finder
    return decide_sync(loaded, {"candidates": candidates}, bindings).record


def warnings_of(record: Record) -> list[str]:
    (pick,) = [
        i for i, c in enumerate(record.log) if c["item"] == "call" and "pick" in c["sources"]
    ]
    calls = json.loads(record.document)["unhashed"]["calls"]
    return list(calls[str(pick)].get("warnings", []))


def test_from_record_answers_a_candidate_valued_question_over_the_stored_set() -> None:
    stored = run(["ana@x.org", "bob@y.org"], Model(top="bob@y.org"))
    live = answered_from(stored, ["ana@x.org", "bob@y.org"])
    assert entries(live)["q:sender"]["value"] == entries(stored)["q:sender"]["value"]


@pytest.mark.parametrize(
    "candidates", [["bob@y.org", "ana@x.org"], ["ana@x.org"], ["ana@x.org", "bob@y.org", "cy"]]
)
def test_from_record_refuses_a_question_over_another_set(candidates: list[str]) -> None:
    stored = run(["ana@x.org", "bob@y.org"])
    live = answered_from(stored, candidates)
    assert marker(live, "q:sender") == ("Unavailable", "unsupported", ["q:sender"])
    assert "value" in entries(live)["q:other"]
    assert "'q:sender': its labels differ from the stored question's" in warnings_of(live)


def test_from_record_answers_over_a_rules_recomputed_set() -> None:
    d = rule_set(found)
    stored = run(["ana@x.org"], d=d)
    live = answered_from(stored, ["ana@x.org"], d)
    assert entries(live)["q:sender"]["value"] == entries(stored)["q:sender"]["value"]


def test_from_record_refuses_a_question_whose_stored_set_is_a_marker() -> None:
    stored = run(["ana@x.org"])
    doc = json.loads(stored.document)
    (at,) = [n for n, i in enumerate(doc["log"]) if i.get("ref") == SET]
    item = {k: v for k, v in doc["log"][at].items() if k != "value"}
    item["marker"] = {"kind": "Unavailable", "cause": {"code": "refused", "refs": [SET]}}
    doc["log"][at] = item
    live = answered_from(read_record(rehashed(doc)), ["ana@x.org"])
    assert marker(live, "q:sender") == ("Unavailable", "unsupported", ["q:sender"])
    line = f"'q:sender': its candidate set '{SET}' is stored as Unavailable refused"
    assert line in warnings_of(live)


@pytest.mark.parametrize("candidates", [["ana@x.org"], [], ["none"]])
def test_a_candidate_valued_turn_replays_with_no_finding(candidates: list[str]) -> None:
    loaded = load(design())
    record = decide_sync(
        loaded, {"candidates": candidates}, {"default": Model(), "tools": finder}
    ).record
    assert replay_sync(record, loaded)[1] == []


# ---------------------------------------------------------------------------------------------
# Replay


def found_v1(r: Any) -> Any:
    return ["ana@x.org", "bob@y.org"]


def found_v2(r: Any) -> Any:
    return ["ana@x.org", "cy@z.org"]


def found_reversed(r: Any) -> Any:
    return ["bob@y.org", "ana@x.org"]


def versioned(function: Any, version: str) -> Design:
    d = rule_set(function)
    rule = d.rules["find"]
    assert isinstance(rule, Rule)
    return Design(
        name=d.name,
        state=d.state,
        sources=d.sources,
        rules={
            "find": Rule(
                rule.provides,
                rule.outputs,
                function,
                version=version,
                reads=list(rule.reads),
                candidates=True,
            )
        },
        policy=d.policy,
    )


@pytest.mark.parametrize("changed", [found_v2, found_reversed])
def test_a_recomputed_set_that_differs_is_question_changed(changed: Any) -> None:
    recorded = run(["x"], Model(top="ana@x.org"), versioned(found_v1, "1"))
    _, found = replay_sync(recorded, load(versioned(changed, "2")))
    kinds = [f["kind"] for f in found]
    assert kinds[:2] == ["design-mismatch", "question-changed"]
    assert found[1]["refs"] == ["q:sender"]
    (entry,) = [f for f in found if f["kind"] == "entry" and f["ref"] == "d:find"]
    assert entry["rule"]["version"] == "2"


def test_a_recomputed_set_alike_is_no_question_changed() -> None:
    recorded = run(["x"], Model(), versioned(found_v1, "1"))
    _, found = replay_sync(recorded, load(versioned(found_v1, "2")))
    assert "question-changed" not in [f["kind"] for f in found]


def test_another_records_set_answering_is_no_question_changed() -> None:
    from ommat.evaluation import use_answers_sync

    loaded = load(design())
    first = run(["ana@x.org", "bob@y.org"], Model(top="ana@x.org"))

    async def other_set(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        return CallResult({SET: ["bob@y.org", "cy@z.org"]})

    second = run(["ana@x.org", "bob@y.org"], Model(top="bob@y.org"), tools=other_set)
    report = use_answers_sync(first, loaded, second)
    kinds = [f["kind"] for f in report["findings"]]
    assert "question-changed" not in kinds
    refs = [f["ref"] for f in report["findings"] if f["kind"] == "entry"]
    assert SET in refs


# ---------------------------------------------------------------------------------------------
# The redacted view

ROOT = bytes(range(32))


def digest_of(record: Record, value: Any) -> str:
    """The keyed digest a verifier recomputes: HMAC-SHA-256 under the record's per-turn key
    over the value's RFC 8785 bytes."""
    import hashlib
    import hmac

    import rfc8785

    from ommat._canonical import turn_key

    key = turn_key(ROOT, record.record_hash)
    return "hmac-sha256:" + hmac.new(key, rfc8785.dumps(value), hashlib.sha256).hexdigest()


def redacted_entries(record: Record) -> dict[str, Any]:
    from ommat import redact

    view = json.loads(redact(record, ROOT))
    return {i["ref"]: i for i in view["log"] if i["item"] == "entry"}


CANDIDATES = ["ana@x.org", "Zoë Ångström"]


def test_a_digested_set_digests_its_candidates_in_the_answer_too() -> None:
    record = run(CANDIDATES, Model(top="ana@x.org"))
    view = redacted_entries(record)
    assert view[SET]["value"] == digest_of(record, CANDIDATES)
    answer = view["q:sender"]["value"]
    digests = [digest_of(record, c) for c in CANDIDATES]
    assert answer["labels"] == [*FIXED, *digests]
    original = entries(record)["q:sender"]["value"]["probabilities"]
    assert answer["probabilities"] == {
        **{f: original[f] for f in FIXED},
        **{digest_of(record, c): original[c] for c in CANDIDATES},
    }
    from ommat import redact

    text = redact(record, ROOT).decode("utf-8")
    for c in CANDIDATES:
        assert json.dumps(c, ensure_ascii=False)[1:-1] not in text


def test_an_open_set_shows_its_candidates_in_the_set_and_the_answer() -> None:
    d = design()
    d = Design(
        name=d.name,
        state=d.state,
        sources=d.sources,
        policy=d.policy,
        classes={SET: "open"},
    )
    record = run(CANDIDATES, Model(), d)
    view = redacted_entries(record)
    assert view[SET]["value"] == CANDIDATES
    assert view["q:sender"]["value"]["labels"] == [*FIXED, *CANDIDATES]


def test_a_question_left_out_keeps_its_marker_in_the_view() -> None:
    record = run([])
    assert redacted_entries(record)["q:sender"]["marker"]["cause"]["code"] == "when-false"


# ---------------------------------------------------------------------------------------------
# The lints


def picked(r: Any) -> Any:
    from ommat.helpers import pick_with_margin

    return pick_with_margin(r, "q:sender", 0.6, 0.2)


def mark_if_named(r: Any) -> Plan:
    from ommat import Undetermined

    sender = r.read("d:sender")
    if isinstance(sender, Undetermined) or sender in FIXED:
        return Plan("clarify", reply="Who?")
    return Plan("inform", reply="named")


def lint_design(*, pick_reads_other: bool = False) -> Design:
    finder = Source({SET: {"type": t.list_(t.string), "candidates": True}}, needs=["t:state"])
    return Design(
        name="lint",
        state=t.record(),
        sources={"finder": finder, "pick": asks()},
        rules={
            "sender": Rule("d:sender", t.string, picked, version="1", reads=["q:sender"]),
        },
        policy=Policy(mark_if_named, version="1", reads=["d:sender"]),
    )


SETS = {SET: [[], ["ana@x.org"], ["none"]]}


def test_the_dead_lint_answers_a_candidate_valued_question_over_its_rendered_labels() -> None:
    from ommat.helpers import dead_questions

    report = dead_questions(load(lint_design()), values=SETS)
    sender = report["questions"]["q:sender"]
    assert sender["status"] == "live"
    witness = sender["witness"]
    # Each side of the witness shows the answer as its run wrote it, over rendered labels.
    labels = [w["q:sender"].get("labels") for w in witness]
    assert ["none", "several", "ana@x.org"] in labels
    assert report["questions"]["q:other"]["status"] == "not-on-grid"


def test_the_dead_lint_writes_no_invalid_answer_for_a_candidate_valued_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ommat.helpers._dead as dead

    seen: list[Any] = []
    real = dead._run

    def spy(loaded: Any, point: Any, keys: Any) -> Any:
        out = real(loaded, point, keys)
        seen.append(out[1])
        return out

    monkeypatch.setattr(dead, "_run", spy)
    dead.dead_questions(load(lint_design()), values=SETS)
    codes = {e["cause"]["code"] for e in (s.get("q:sender", {}) for s in seen) if "cause" in e}
    assert "invalid-answer" not in codes
    assert {"when-false", "label-collision"} <= codes


def test_a_candidate_valued_question_that_changes_nothing_is_not_on_grid() -> None:
    from ommat.helpers import dead_questions

    d = lint_design()
    d = Design(name=d.name, state=d.state, sources=d.sources, policy=Policy(inform, version="1"))
    report = dead_questions(load(d), values=SETS)
    assert report["questions"]["q:sender"] == {"status": "not-on-grid", "witness": None}


def test_a_candidate_set_with_no_values_is_unfilled_before_any_run() -> None:
    from ommat.helpers import dead_questions

    with pytest.raises(ValueError, match=SET):
        dead_questions(load(lint_design()))


def test_the_unsure_lint_runs_over_a_candidate_valued_design() -> None:
    from ommat.helpers import unsure_to_execute

    report = unsure_to_execute(load(lint_design()), values={"d:sender": ["ana@x.org", "none"]})
    assert report["execute"] == []


# ---------------------------------------------------------------------------------------------
# The binding conformance suite over a candidate-valued turn's batches


@pytest.mark.parametrize("candidates", [["ana@example.com", '"Ana" <a@x.org>'], ["]", "李"]])
def test_from_record_and_the_fixture_pass_the_suite_on_the_batches_their_record_answers(
    candidates: list[str],
) -> None:
    from ommat._fixture import FixtureBinding, RecordedCalls
    from ommat.bindings.conformance import check

    model = Model()
    loaded = load(design())
    record = decide_sync(
        loaded, {"candidates": candidates}, {"default": model, "tools": finder}
    ).record
    seen = list(model.batches)
    assert check(from_record(record, loaded), batches=seen) == []
    assert check(FixtureBinding(RecordedCalls(record.log), loaded), batches=seen) == []
