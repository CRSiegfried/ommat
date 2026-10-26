"""The per-candidate picks: ``pick_each_with_margin`` and ``pick_each_with_thresholds`` over a
per-candidate question's declared ref, each candidate's pick, abstention or marker in the set's
order; the record a rule passing one candidate's abstention on writes, a flagged context entry
included; and the board check on a returned cause naming a grounded ref."""

import json
from collections.abc import Callable
from typing import Any, cast

import pytest

from designs import slot
from ledger_logs import rehashed
from ommat import (
    CallResult,
    Cause,
    CauseCode,
    Context,
    ContextEntry,
    Design,
    LoadedDesign,
    Plan,
    Policy,
    Question,
    ReadView,
    Record,
    Rule,
    Source,
    SourceCall,
    Text,
    Unavailable,
    Unsure,
    decide_sync,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat._refs import grounded_ref
from ommat.helpers import determined, pick_each_with_margin, pick_each_with_thresholds

OUT = "q:lamps/out"
SET = "d:streets"
STREETS = ["Quay Street", "Elm Row", "Bridge Lane", "Back Lane"]

PICKS: list[object] = []


def streets(r: ReadView) -> object:
    names = r.read("t:state.streets")
    if names == ["raise"]:
        raise RuntimeError("no streets")
    return names


def lit(r: ReadView) -> object:
    picks = pick_each_with_margin(r, OUT, 0.8, 0.2)
    PICKS.append(picks)
    if not isinstance(picks, dict):
        return picks
    return [street for street, pick in picks.items() if pick == "yes"]


def first_doubt(r: ReadView) -> object:
    """The first street's pick, its abstention passed on whole."""
    picks = pick_each_with_thresholds(r, OUT, {"yes": 0.8, "no": 0.8}, 0.2)
    PICKS.append(picks)
    if not isinstance(picks, dict):
        return picks
    if not picks:
        return "no"
    return next(iter(picks.values()))


def both(r: ReadView) -> object:
    got = determined(r, "d:lit", "d:doubt")
    if not isinstance(got, tuple):
        return got
    return "yes" if got[0] else "no"


def escalate(r: ReadView) -> Plan:
    return Plan("escalate", reason="checked")


SOURCE = Source(
    {OUT: Question("noul", Text("Is a lamp out on {s}?", [slot("s", candidate=SET)]))},
    needs=["t:state.text"],
    grounded=SET,
)


def design(
    lit_rule: Callable[[ReadView], object] = lit,
    doubt_rule: Callable[[ReadView], object] = first_doubt,
    lit_reads: tuple[str, ...] = (OUT,),
) -> Design:
    return Design(
        name="lamps",
        state=t.record(text=t.string, streets=t.list_(t.string)),
        sources={"lamps": SOURCE},
        rules={
            "streets": Rule(
                SET,
                t.list_(t.string),
                streets,
                version="1",
                reads=["t:state.streets"],
                candidates=True,
            ),
            "lit": Rule("d:lit", t.list_(t.string), lit_rule, version="1", reads=list(lit_reads)),
            "doubt": Rule("d:doubt", t.label("yes", "no"), doubt_rule, version="1", reads=[OUT]),
            "both": Rule(
                "d:both", t.label("yes", "no"), both, version="1", reads=["d:lit", "d:doubt"]
            ),
        },
        policy=Policy(escalate, version="1", reads=["d:both"]),
        context=Context([ContextEntry("d:doubt", Text("Doubt: {d}", [slot("d", ref="d:doubt")]))]),
    )


LOADED = load(design())

ANSWERS: dict[str, dict[str, float]] = {
    "Quay Street": {"yes": 0.95, "no": 0.05},
    "Elm Row": {"yes": 0.55, "no": 0.45},
    "Bridge Lane": {"yes": 0.05, "no": 0.95},
}
"""Back Lane has no answer, so its entry is a marker."""


async def answer(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    asked = {ref for s in batch for ref in s.questions}
    out: dict[str, Any] = {}
    for street, probabilities in ANSWERS.items():
        if grounded_ref(OUT, street) in asked:
            out[grounded_ref(OUT, street)] = dict(probabilities)
    return CallResult(out)


def decided(names: list[str], loaded: LoadedDesign = LOADED) -> Record:
    PICKS.clear()
    state = {"text": "Lamps out", "streets": names}
    return decide_sync(loaded, state, {"default": answer}).record


def entries(record: Record) -> dict[str, dict[str, Any]]:
    return {i["ref"]: i for i in record.log if i["item"] == "entry"}


def test_each_candidate_gets_its_pick_its_abstention_or_its_marker_in_the_set_s_order() -> None:
    record = decided(["Elm Row", "Quay Street", "Back Lane", "Bridge Lane"])
    held, _ = PICKS
    assert isinstance(held, dict)
    picks = cast("dict[str, object]", held)
    assert list(picks) == ["Elm Row", "Quay Street", "Back Lane", "Bridge Lane"]
    assert picks["Quay Street"] == "yes"
    assert picks["Bridge Lane"] == "no"
    elm = picks["Elm Row"]
    assert isinstance(elm, Unsure)
    assert elm.cause.code.value == "abstained"
    assert list(elm.cause.refs) == [grounded_ref(OUT, "Elm Row")]
    assert isinstance(picks["Back Lane"], Unavailable)
    assert entries(record)["d:lit"]["value"] == ["Quay Street"]


def test_the_derivation_names_the_declared_ref_alone() -> None:
    record = decided(STREETS)
    (call,) = entries(record)["d:lit"]["derivation"]
    assert call == {"helper": "pick_each_with_margin", "refs": [OUT], "rested_on": [OUT]}
    (call,) = entries(record)["d:both"]["derivation"]
    assert call == {
        "helper": "determined",
        "refs": ["d:lit", "d:doubt"],
        "rested_on": ["d:lit", "d:doubt"],
    }


def test_a_rule_passing_one_candidate_s_abstention_on_into_the_context_records_cleanly() -> None:
    record = decided(["Elm Row", "Quay Street"])
    doubt = entries(record)["d:doubt"]
    assert doubt["marker"]["cause"] == {
        "code": "abstained",
        "refs": [grounded_ref(OUT, "Elm Row")],
    }
    (plan,) = [i for i in record.log if i["item"] == "plan"]
    (context_entry,) = plan["context"]["entries"]
    assert context_entry["marker"] == doubt["marker"]
    # The record reads back, replays and redacts with the candidate digested in the cause.
    stored = read_record(record.document)
    assert replay_sync(stored, LOADED)[1] == []
    redacted = redact(record, b"k" * 32).decode("utf-8")
    assert "Elm Row" not in redacted


def test_redact_digests_a_grounded_ref_in_a_plan_s_own_cause() -> None:
    doc = json.loads(decided(["Elm Row", "Quay Street"]).document)
    plan = next(i for i in doc["log"] if i["item"] == "plan")
    plan["cause"] = {"code": "required", "refs": [grounded_ref(OUT, "Elm Row")]}
    record = read_record(json.dumps(rehashed(doc)).encode("utf-8"))
    redacted = json.loads(redact(record, b"k" * 32))
    (cause_ref,) = next(i for i in redacted["log"] if i["item"] == "plan")["cause"]["refs"]
    assert cause_ref.startswith('q:lamps/out["hmac-sha256:')
    assert "Elm Row" not in json.dumps(redacted)


def forged(r: ReadView) -> object:
    """A cause naming a candidate the set never held."""
    return Unsure(Cause(CauseCode.abstained, [grounded_ref(OUT, "Nowhere Lane")]))


def on_board(r: ReadView) -> object:
    """A cause naming a candidate the turn asked about."""
    return Unsure(Cause(CauseCode.abstained, [grounded_ref(OUT, "Quay Street")]))


def test_a_returned_cause_names_a_grounded_ref_only_when_the_turn_wrote_its_entry() -> None:
    kept = entries(decided(["Quay Street"], load(design(doubt_rule=on_board))))["d:doubt"]
    assert kept["marker"]["cause"]["refs"] == [grounded_ref(OUT, "Quay Street")]
    refused = entries(decided(["Quay Street"], load(design(doubt_rule=forged))))["d:doubt"]
    assert refused["marker"]["cause"] == {"code": "invalid-return", "refs": ["doubt"]}


def test_the_set_s_marker_comes_back_unchanged_from_both_picks() -> None:
    record = decided(["raise"])
    for picks in PICKS:
        assert isinstance(picks, Unavailable)
        assert picks.cause.code.value == "need-unavailable"
    assert entries(record)["d:lit"]["marker"] == entries(record)[OUT]["marker"]


def test_an_empty_set_gives_an_empty_dict_from_both_picks() -> None:
    record = decided([])
    assert PICKS == [{}, {}]
    assert entries(record)["d:lit"]["value"] == []


def lacking(r: ReadView) -> object:
    return pick_each_with_thresholds(r, OUT, {"maybe": 0.5}, 0.0)


def bad_bar(r: ReadView) -> object:
    return pick_each_with_margin(r, OUT, 1.5, 0.0)


def bad_margin(r: ReadView) -> object:
    return pick_each_with_thresholds(r, OUT, {"yes": 0.5}, -0.1)


def not_per_candidate(r: ReadView) -> object:
    return pick_each_with_margin(r, SET, 0.8, 0.2)


@pytest.mark.parametrize(
    ("rule", "names"),
    [
        (lacking, ["Back Lane", "Quay Street", "Elm Row"]),
        (bad_bar, ["Quay Street"]),
        (bad_margin, ["Quay Street"]),
        (bad_bar, []),
        (not_per_candidate, ["Quay Street"]),
    ],
    ids=["a label every answer lacks", "bar", "margin", "bar over an empty set", "a list read"],
)
def test_a_misuse_raises_and_the_rule_s_entry_says_so(
    rule: Callable[[ReadView], object], names: list[str]
) -> None:
    # The lacking label raises with a marker first among the candidates, before any pick.
    reads = (OUT, SET) if rule is not_per_candidate else (OUT,)
    loaded = load(design(lit_rule=rule, lit_reads=reads))
    marker = entries(decided(names, loaded))["d:lit"]["marker"]
    assert marker["cause"] == {"code": "raised", "refs": ["lit"]}


# ---------------------------------------------------------------------------------------------
# The helpers over a read view, each candidate's entry set by hand

from ommat import Answer  # noqa: E402
from ommat._read_view import ComponentKind, Role  # noqa: E402

GONE = Unavailable(Cause(CauseCode.raised, ["lamps"]))


def board_view(entries: dict[str, object], *, marker: object | None = None) -> ReadView:
    board: dict[str, object] = {"t:state": {}, SET: list(entries)}
    for street, entry in entries.items():
        board[grounded_ref(OUT, street)] = entry
    if marker is not None:
        board[OUT] = marker
    return ReadView(Role(ComponentKind.rule, "lit"), (OUT, SET), (), board, {}, grounded={OUT: SET})


def noul(yes: float) -> Answer:
    return Answer("noul", ["yes", "no"], {"yes": yes, "no": round(1 - yes, 2)})


def choice(a: float, b: float, c: float) -> Answer:
    return Answer("choice", ["a", "b", "c"], {"a": a, "b": b, "c": c})


def test_the_bar_alone_and_the_margin_alone_each_reject_a_candidate() -> None:
    # 0.79 leads by 0.58 but misses a bar of 0.8; 0.85 clears any bar but leads by only 0.75.
    yes_no = board_view({"Low": noul(0.79), "Clear": noul(0.95)})
    abc = board_view({"Close": choice(0.85, 0.1, 0.05), "Sure": choice(0.97, 0.02, 0.01)})
    by_bar = cast("dict[str, object]", pick_each_with_margin(yes_no, OUT, 0.8, 0.0))
    assert isinstance(by_bar["Low"], Unsure)
    assert by_bar["Clear"] == "yes"
    by_margin = cast("dict[str, object]", pick_each_with_margin(abc, OUT, 0.0, 0.8))
    assert isinstance(by_margin["Close"], Unsure)
    assert by_margin["Sure"] == "a"
    by_bar = cast("dict[str, object]", pick_each_with_thresholds(yes_no, OUT, {"yes": 0.8}, 0.0))
    assert isinstance(by_bar["Low"], Unsure)
    assert by_bar["Clear"] == "yes"
    by_margin = cast("dict[str, object]", pick_each_with_thresholds(abc, OUT, {"a": 0.0}, 0.8))
    assert isinstance(by_margin["Close"], Unsure)
    assert by_margin["Sure"] == "a"


BOTH: list[Callable[[ReadView], object]] = [
    lambda r: pick_each_with_margin(r, OUT, 0.8, 0.2),
    lambda r: pick_each_with_thresholds(r, OUT, {"yes": 0.8, "no": 0.8}, 0.2),
]


@pytest.mark.parametrize("pick", BOTH, ids=["margin", "thresholds"])
def test_a_new_dict_in_the_set_s_order_each_marker_the_entry_itself(
    pick: Callable[[ReadView], object],
) -> None:
    entries: dict[str, object] = {"Zed": noul(0.9), "Gone": GONE, "Abel": noul(0.1)}
    r = board_view(entries)
    read = r.read(OUT)
    picks = cast("dict[str, object]", pick(r))
    assert list(picks) == ["Zed", "Gone", "Abel"]
    assert picks["Gone"] is GONE
    assert picks == {"Zed": "yes", "Gone": GONE, "Abel": "no"}
    assert picks is not read
    assert r.read(OUT) == read


@pytest.mark.parametrize(
    ("pick", "name"),
    list(zip(BOTH, ["pick_each_with_margin", "pick_each_with_thresholds"], strict=True)),
    ids=["margin", "thresholds"],
)
def test_the_set_s_marker_comes_back_itself_and_the_call_rests_on_the_ref(
    pick: Callable[[ReadView], object], name: str
) -> None:
    need = Unavailable(Cause(CauseCode.need_unavailable, [SET]))
    r = board_view({}, marker=need)
    assert pick(r) is need
    assert [(i.helper, i.refs, i.rested_on) for i in r.helper_log()] == [(name, (OUT,), (OUT,))]


BAD_ARGUMENTS: list[tuple[Callable[[ReadView], object], str]] = [
    (lambda r: pick_each_with_margin(r, OUT, 1.5, 0.2), "bar is not a finite number"),
    (lambda r: pick_each_with_margin(r, OUT, 0.8, -0.1), "margin is not a finite number"),
    (lambda r: pick_each_with_margin(r, OUT, True, 0.2), "bar is not exactly an int or a float"),
    (lambda r: pick_each_with_thresholds(r, OUT, [0.5], 0.2), "thresholds is not exactly a dict"),  # type: ignore[arg-type]
    (lambda r: pick_each_with_thresholds(r, OUT, {1: 0.5}, 0.2), "a key that is not exactly a str"),  # type: ignore[dict-item]
    (lambda r: pick_each_with_thresholds(r, OUT, {"yes": 2}, 0.2), "thresholds is not a finite"),
    (lambda r: pick_each_with_thresholds(r, OUT, {"yes": 0.5}, 1.5), "margin is not a finite"),
]


@pytest.mark.parametrize("over_a_marker", [False, True], ids=["over answers", "over a marker"])
@pytest.mark.parametrize(("pick", "message"), BAD_ARGUMENTS)
def test_each_bad_argument_raises_even_over_a_marker(
    pick: Callable[[ReadView], object], message: str, over_a_marker: bool
) -> None:
    marker = Unsure(Cause(CauseCode.need_undetermined, [SET])) if over_a_marker else None
    r = board_view({"Quay Street": noul(0.9)}, marker=marker)
    with pytest.raises((TypeError, ValueError), match=message):
        pick(r)


def test_a_read_that_is_not_per_candidate_or_an_entry_that_is_not_an_answer_raises() -> None:
    r = board_view({"Quay Street": noul(0.9), "Odd": 3, "Elm Row": noul(0.1)})
    with pytest.raises(TypeError, match=r"Odd.*which holds no answer"):
        pick_each_with_margin(r, OUT, 0.8, 0.2)
    with pytest.raises(TypeError, match="is not a per-candidate question's declared ref"):
        pick_each_with_margin(r, SET, 0.8, 0.2)


def test_thresholds_pick_the_most_probable_qualifier_when_it_is_not_the_first_label() -> None:
    r = board_view({"Late": choice(0.1, 0.6, 0.3)})
    picks = cast("dict[str, object]", pick_each_with_thresholds(r, OUT, {"a": 0.0, "b": 0.1}, 0.0))
    assert picks == {"Late": "b"}


@pytest.mark.parametrize("pick", BOTH, ids=["margin", "thresholds"])
def test_an_empty_set_gives_a_new_empty_dict(pick: Callable[[ReadView], object]) -> None:
    r = board_view({})
    read = r.read(OUT)
    got = pick(r)
    assert got == {}
    assert got is not read
