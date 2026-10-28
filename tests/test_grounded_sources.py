# pyright: reportPrivateUsage=false
"""Grounded sources of ``t:`` refs and slots on grounded entries: a source grounded over a
candidate set provides each ``t:`` ref once per candidate, under the grounded refs the engine
forms, each value checked and written alone; a per-candidate question's slot on a declared ref
grounded over its own set is filled from that candidate's entry, is that question's need alone
under its grounded ref, and a marker there leaves that question out. ``candidate_of``, the
record, replay, ``from_record`` and redaction over them."""

import copy
import json
from typing import Any

import pytest

from designs import slot
from ledger_logs import rehashed
from ommat import (
    CallResult,
    Cause,
    CauseCode,
    Design,
    InvalidDesign,
    LoadedDesign,
    Plan,
    Policy,
    Question,
    ReadView,
    Record,
    Rule,
    RunSettings,
    Source,
    SourceCall,
    Text,
    Unavailable,
    decide_sync,
    from_record,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat._refs import grounded_ref
from ommat.bindings import candidate_of
from test_views import ROOT, digest

PERSON = t.record(name=t.string, age=t.integer())
SET = "d:people"
CITY = "t:city"
INFO = "t:info"
OUT = "q:ask/out"

ANA: dict[str, Any] = {"name": "Ana", "age": 30}
BO: dict[str, Any] = {"name": "Bo", "age": 41}


def people(r: ReadView) -> object:
    value = r.read("t:state.people")
    if value == [{"name": "raise", "age": 0}]:
        raise RuntimeError("no people")
    return value


READ: list[object] = []


def cities(r: ReadView) -> object:
    READ.append(r.read(CITY))
    return "seen"


def plan(r: ReadView) -> Plan:
    return Plan("inform", reply="done")


def design(**changes: Any) -> Design:
    fields: dict[str, Any] = {
        "name": "facts",
        "state": t.record(people=t.list_(PERSON)),
        "sources": {
            "lookup": Source(
                {CITY: t.nullable(t.string), INFO: t.record(city=t.string)},
                needs=[SET],
                oracle="tools",
                grounded=SET,
            ),
            "ask": Source(
                {
                    OUT: Question(
                        "noul",
                        Text(
                            "Does {c} live in {city} ({again})?",
                            [
                                slot("c", candidate=SET),
                                slot("city", ref=CITY),
                                slot("again", ref="t:info.city"),
                            ],
                        ),
                    )
                },
                grounded=SET,
            ),
        },
        "rules": {
            "people": Rule(
                SET,
                t.list_(PERSON),
                people,
                version="1",
                reads=["t:state.people"],
                candidates="name",
            ),
            "cities": Rule("d:cities", t.string, cities, version="1", reads=[CITY]),
        },
        "policy": Policy(plan, version="1", reads=["d:cities"]),
        "classes": {SET: {"name": "identifier"}},
    }
    fields.update(changes)
    return Design(**fields)


LOADED = load(design())

TOOLS: list[SourceCall] = []
ASKED: list[SourceCall] = []
CITIES: dict[str, object] = {"Ana": "Oslo", "Bo": "Rome"}
EXTRA: dict[str, object] = {}


async def tools(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    (source,) = batch
    TOOLS.append(source)
    members = {m["name"]: m for m in source.needs[SET]}
    values: dict[str, object] = {}
    for ref in source.refs:
        name = candidate_of(ref)
        assert name in members
        if ref.startswith(CITY) and name in CITIES:
            values[ref] = CITIES[name]
        elif ref.startswith(INFO):
            values[ref] = {"city": f"{name}-town"}
    values.update(EXTRA)
    return CallResult(values)


async def answer(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    ASKED.extend(batch)
    return CallResult({ref: {"yes": 0.9, "no": 0.1} for s in batch for ref in s.questions})


def decided(
    members: list[Any] | None = None,
    loaded: LoadedDesign = LOADED,
    settings: RunSettings | None = None,
    **bindings: Any,
) -> Record:
    TOOLS.clear()
    ASKED.clear()
    READ.clear()
    state = {"people": [BO, ANA] if members is None else members}
    given: dict[str, Any] = {"default": answer, "tools": tools, **bindings}
    return decide_sync(loaded, state, given, settings).record


def entries(record: Record) -> dict[str, dict[str, Any]]:
    return {i["ref"]: i for i in record.log if i["item"] == "entry"}


def marker(item: dict[str, Any]) -> tuple[str, str, list[str]]:
    m = item["marker"]
    return m["kind"], m["cause"]["code"], m["cause"]["refs"]


@pytest.fixture(autouse=True)
def _reset() -> Any:
    yield
    CITIES.clear()
    CITIES.update({"Ana": "Oslo", "Bo": "Rome"})
    EXTRA.clear()


# ---------------------------------------------------------------------------------------------
# Load


def test_a_grounded_t_source_loads_with_its_refs_grounded() -> None:
    assert dict(LOADED._grounded) == {CITY: SET, INFO: SET, OUT: SET}
    assert LOADED._graph["lookup"] == frozenset({"people"})
    assert LOADED._graph["ask"] == frozenset({"lookup", "people"})
    assert LOADED._needs["ask"] == frozenset()  # a grounded slot is each candidate's need
    assert LOADED._needs["lookup"] == frozenset({SET})


def test_check_4_refuses_a_slot_on_a_ref_grounded_over_another_set() -> None:
    def names(r: ReadView) -> object:
        return ["x"]

    other = Rule("d:other", t.list_(t.string), names, version="1", candidates=True)
    ask = Source(
        {
            OUT: Question(
                "noul",
                Text("{c} in {city}?", [slot("c", candidate="d:other"), slot("city", ref=CITY)]),
            )
        },
        grounded="d:other",
    )
    d = design(rules={**design().rules, "other": other}, sources={**design().sources, "ask": ask})
    with pytest.raises(InvalidDesign) as info:
        load(d)
    assert [(f.check, f.message) for f in info.value.failures] == [
        (
            4,
            'source "ask"\'s question "q:ask/out" has a slot on "t:city", grounded over '
            '"d:people", not its source\'s set',
        )
    ]


def test_check_12_refuses_a_need_on_a_grounded_declared_ref() -> None:
    reader = Source({"q:x": Question("noul", Text("x?"))}, needs=[CITY])
    with pytest.raises(InvalidDesign) as info:
        load(design(sources={**design().sources, "reader": reader}))
    assert [f.check for f in info.value.failures] == [12]
    assert "need or slot ref on a grounded declared ref" in str(info.value)


# ---------------------------------------------------------------------------------------------
# A turn


def test_the_binding_gets_each_candidates_grounded_ref_and_answers_by_candidate_of() -> None:
    record = decided()
    (source,) = TOOLS
    assert source.refs == tuple(
        sorted(grounded_ref(r, n) for r in (CITY, INFO) for n in ("Ana", "Bo"))
    )
    assert source.needs[SET] == [BO, ANA]
    e = entries(record)
    assert e[grounded_ref(CITY, "Ana")]["value"] == "Oslo"
    assert e[grounded_ref(INFO, "Bo")]["value"] == {"city": "Bo-town"}
    assert READ == [{"Bo": "Rome", "Ana": "Oslo"}]


def test_a_per_candidate_question_is_filled_from_its_candidates_grounded_entries() -> None:
    decided()
    (source,) = ASKED
    ana = grounded_ref(OUT, "Ana")
    assert source.questions[ana].wording == "Does Ana live in Oslo (Ana-town)?"
    assert source.needs == {
        grounded_ref(CITY, "Ana"): "Oslo",
        grounded_ref(CITY, "Bo"): "Rome",
        grounded_ref(INFO, "Ana") + ".city": "Ana-town",
        grounded_ref(INFO, "Bo") + ".city": "Bo-town",
    }


def test_a_missing_key_and_a_misfit_value_mark_their_entries_alone() -> None:
    del CITIES["Bo"]
    CITIES["Ana"] = 3
    e = entries(decided())
    assert marker(e[grounded_ref(CITY, "Bo")]) == (
        "Unavailable",
        "not-returned",
        [grounded_ref(CITY, "Bo")],
    )
    assert marker(e[grounded_ref(CITY, "Ana")]) == (
        "Unavailable",
        "invalid-return",
        [grounded_ref(CITY, "Ana")],
    )
    assert e[grounded_ref(INFO, "Ana")]["value"] == {"city": "Ana-town"}


def test_the_declared_ref_as_a_key_is_unexpected_ref() -> None:
    EXTRA[CITY] = "Oslo"
    e = entries(decided())
    assert marker(e[grounded_ref(CITY, "Ana")])[1] == "unexpected-ref"


def test_a_null_slot_value_renders_as_declared_text_does() -> None:
    CITIES["Ana"] = None
    decided()
    (source,) = ASKED
    assert source.questions[grounded_ref(OUT, "Ana")].wording == "Does Ana live in null (Ana-town)?"


def test_a_marker_on_a_grounded_slot_leaves_that_candidates_question_out() -> None:
    del CITIES["Bo"]
    record = decided()
    e = entries(record)
    bo = grounded_ref(OUT, "Bo")
    assert marker(e[bo]) == ("Unavailable", "need-unavailable", [grounded_ref(CITY, "Bo")])
    assert e[bo]["call"] is None
    (source,) = ASKED
    assert list(source.questions) == [grounded_ref(OUT, "Ana")]


def test_a_set_holding_a_marker_marks_each_declared_ref_and_calls_no_source() -> None:
    e = entries(decided([{"name": "raise", "age": 0}]))
    for ref in (CITY, INFO, OUT):
        assert marker(e[ref]) == ("Unavailable", "need-unavailable", [SET])
    assert TOOLS == []
    assert ASKED == []


def test_an_empty_set_calls_no_source() -> None:
    record = decided([])
    assert TOOLS == []
    assert ASKED == []
    assert not any(i["item"] == "call" for i in record.log)
    assert READ == [{}]


def test_a_source_marker_goes_on_each_grounded_ref() -> None:
    async def refusing(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        return CallResult(
            {ref: Unavailable(Cause(CauseCode.refused)) for s in batch for ref in s.refs}
        )

    e = entries(decided(tools=refusing))
    assert marker(e[grounded_ref(CITY, "Ana")]) == (
        "Unavailable",
        "refused",
        [grounded_ref(CITY, "Ana")],
    )
    assert marker(e[grounded_ref(OUT, "Ana")])[1] == "need-unavailable"


def test_the_round_bound_cuts_a_grounded_source_before_it_is_asked() -> None:
    record = decided(settings=RunSettings(round_bound=1))
    e = entries(record)
    assert e[grounded_ref(CITY, "Ana")]["value"] == "Oslo"
    assert marker(e[grounded_ref(OUT, "Ana")]) == ("Unasked", "round-bound", [])


# ---------------------------------------------------------------------------------------------
# candidate_of


@pytest.mark.parametrize(
    ("ref", "candidate"),
    [
        ('t:city["Ana"]', "Ana"),
        ('t:city["Zoë"].x', "Zoë"),
        ('t:city["Zo\\u00eb"]', None),
        ("t:city", None),
        ('t:city["Ana"', None),
        ("nonsense", None),
        (3, None),
    ],
)
def test_candidate_of_reads_a_grounded_refs_candidate(ref: object, candidate: str | None) -> None:
    assert candidate_of(ref) == candidate


# ---------------------------------------------------------------------------------------------
# The record


def test_the_settings_item_names_each_grounded_t_refs_set() -> None:
    assert decided().log[1]["grounded"] == {CITY: SET, INFO: SET, OUT: SET}


def test_a_grounded_source_turn_replays_with_no_finding() -> None:
    record = decided()
    rerun, found = replay_sync(record, LOADED)
    assert found == []
    assert rerun is not None
    assert rerun.record_hash == record.record_hash


def test_from_record_answers_grounded_t_refs_and_slots() -> None:
    record = decided()
    again = decided(tools=from_record(record, LOADED), default=from_record(record, LOADED))
    assert [i for i in again.log if i["item"] == "entry"] == [
        i for i in record.log if i["item"] == "entry"
    ]


def test_from_record_refuses_a_source_whose_grounded_slot_need_differs() -> None:
    record = decided()
    CITIES["Ana"] = "Bergen"
    again = decided(default=from_record(record, LOADED))
    e = entries(again)
    assert marker(e[grounded_ref(OUT, "Ana")])[1] == "unsupported"
    assert marker(e[grounded_ref(OUT, "Bo")])[1] == "unsupported"


def test_redact_digests_a_grounded_t_refs_candidate() -> None:
    record = decided()
    view = json.loads(redact(record, ROOT))
    written = [i["ref"] for i in view["log"] if i["item"] == "entry"]
    assert grounded_ref(CITY, digest("Ana", record)) in written
    assert grounded_ref(CITY, "Ana") not in written


def test_a_record_holding_a_grounded_t_ref_reads_back() -> None:
    record = decided()
    doc = {"log": copy.deepcopy(record.log), "unhashed": copy.deepcopy(record.unhashed)}
    assert read_record(rehashed(doc)).record_hash == record.record_hash
