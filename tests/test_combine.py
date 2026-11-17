"""Combining designs: ``combine`` unions parts into one design under a top policy, each port
bound to a rule or a ref, and refuses what load cannot see; ``standalone`` loads one part
alone."""

import dataclasses
from typing import Any

import pytest

from ommat import (
    Design,
    Group,
    Helper,
    Input,
    Parameter,
    Question,
    ReadView,
    Rule,
    Source,
    Table,
    Target,
    Text,
    Undetermined,
    Unsure,
    decide_sync,
    load,
    replay_sync,
)
from ommat import types as t
from ommat.bindings import fixed
from ommat.helpers import (
    CombineFailure,
    InvalidCombination,
    Part,
    combine,
    dead_questions,
    points,
    standalone,
)

STATE = t.record(message=t.string)
ROUTES = ("email", "calendar", "other")
YES_NO = t.label("yes", "no")
BAR = t.number(0, 1)


def _pick(ref: str, outputs: str, at: str) -> Table:
    """A data rule picking the answer at ``ref`` by the bar and margin under ``at``."""
    return Table(
        "d:" + ref.removeprefix("q:"),
        outputs,
        "F",
        inputs=[Helper("pick_with_margin", ref, bar=f"{at}/bar", margin=f"{at}/margin")],
        rows=[["-", Input(ref)]],
    )


def _bars(at: str) -> dict[str, Parameter]:
    return {f"{at}/bar": Parameter(BAR, 0.6), f"{at}/margin": Parameter(BAR, 0.2)}


def _ask(ref: str, wording: str, labels: Any = None) -> Source:
    kind = "noul" if labels is None else "choice"
    return Source({ref: Question(kind, Text(wording), labels)}, needs=["t:state"])


SELECTOR = Design(
    name="selector",
    state=STATE,
    sources={"router": _ask("q:route", "Which workflow handles the message?", ROUTES)},
    rules={
        "route": _pick("q:route", t.label(*ROUTES), "route"),
        "is_calendar": Table(
            "d:is_calendar", t.boolean, "F", ["d:route"], [["calendar", True], ["Known", False]]
        ),
    },
    parameters=_bars("route"),
    policy="d:route",
)
"""The workflow selector: it picks a workflow, and says whether it picked the calendar."""


def _workflow(name: str, outcomes: tuple[str, str]) -> Part:
    """A workflow reading ``d:<name>/chosen``, a port, beside its own question."""
    chosen, asked = f"d:{name}/chosen", f"q:{name}/sure"
    design = Design(
        name=name,
        state=STATE,
        sources={f"{name}/ask": _ask(asked, f"Is the {name} request clear?")},
        rules={
            f"{name}/sure": _pick(asked, YES_NO, name),
            f"{name}/outcome": Table(
                f"d:{name}/outcome",
                t.label("skip", *outcomes),
                "F",
                [chosen, f"d:{name}/sure"],
                [[False, "-", "skip"], [True, "yes", outcomes[0]], [True, "no", outcomes[1]]],
            ),
        },
        parameters=_bars(name),
        policy=f"d:{name}/outcome",
    )
    return Part(design, {chosen: t.boolean})


EMAIL = _workflow("email", ("file", "ask_folder"))
CALENDAR = _workflow("calendar", ("accept", "ask_time"))
ACTS = ("skip", "file", "ask_folder", "accept", "ask_time")
ACT = Table(
    "d:act",
    t.label(*ACTS),
    "F",
    ["d:route", "d:email/outcome", "d:calendar/outcome"],
    [
        ["email", "-", "-", Input("d:email/outcome")],
        ["calendar", "-", "-", Input("d:calendar/outcome")],
        ["other", "-", "-", "skip"],
    ],
)
"""The top policy: the chosen workflow's outcome."""
EMAIL_CHOSEN = Table(
    "d:email/chosen", t.boolean, "F", ["d:route"], [["email", True], ["Known", False]]
)
BIND: dict[str, Any] = {"d:email/chosen": EMAIL_CHOSEN, "d:calendar/chosen": "d:is_calendar"}


def email_chosen(r: ReadView) -> object:
    """The email port as a function rule: whether the selector picked email."""
    route = r.read("d:route")
    return route if isinstance(route, Undetermined) else route == "email"


def combined(**changes: Any) -> Design:
    fields: dict[str, Any] = {"bind": BIND, "policy": ACT, "name": "assistant"}
    fields.update(changes)
    return combine(Part(SELECTOR), EMAIL, CALENDAR, **fields)


SURE = {"yes": 0.9, "no": 0.1}


def model(route: str, **answers: Any) -> Any:
    probabilities = {r: 0.8 if r == route else 0.1 for r in ROUTES}
    return fixed(
        {"q:route": probabilities, "q:email/sure": SURE, "q:calendar/sure": SURE, **answers}
    )


def decided(design: Design, route: str, **answers: Any) -> Any:
    return decide_sync(load(design), {"message": "hello"}, {"default": model(route, **answers)})


def refused(*parts: Part, **changes: Any) -> list[str]:
    fields: dict[str, Any] = {"bind": BIND, "policy": ACT}
    fields.update(changes)
    with pytest.raises(InvalidCombination) as caught:
        combine(*parts, **fields)
    return [f.message for f in caught.value.failures]


# --- deciding ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "act"), [("email", "file"), ("calendar", "accept"), ("other", "skip")]
)
def test_two_parts_and_a_top_policy_decide(route: str, act: str) -> None:
    decision = decided(combined(), route)
    assert decision.estimate == act
    entries = decision.record.entries
    assert entries["d:email/chosen"]["value"] is (route == "email")
    assert entries["d:calendar/chosen"]["value"] is (route == "calendar")


def test_a_combined_design_keeps_every_declaration_under_its_own_name() -> None:
    design = combined()
    assert design.policy == "d:act"
    assert design.name == "assistant"
    assert design.state == STATE
    assert set(design.sources) == {"router", "email/ask", "calendar/ask"}
    assert set(design.rules) == {
        "route",
        "is_calendar",
        "email/sure",
        "email/outcome",
        "calendar/sure",
        "calendar/outcome",
        "email/chosen",
        "calendar/chosen",
        "act",
    }
    assert design.rules["email/chosen"] is EMAIL_CHOSEN
    assert design.rules["act"] is ACT
    assert set(design.parameters) == {
        f"{p}/{k}" for p in ("email", "calendar", "route") for k in ("bar", "margin")
    }


def test_a_port_bound_to_a_ref_is_a_pass_through_table() -> None:
    rule = combined().rules["calendar/chosen"]
    assert isinstance(rule, Table)
    assert (rule.provides, rule.outputs, rule.hit) == ("d:calendar/chosen", t.boolean, "F")
    assert list(rule.inputs) == ["d:is_calendar"]
    assert list(rule.rows) == [["-", Input("d:is_calendar")]]


def test_a_pass_through_passes_a_marker_on_naming_its_ref() -> None:
    unsure = dict.fromkeys(ROUTES, 1 / 3)
    decision = decided(combined(), "other", **{"q:route": unsure})
    assert isinstance(decision.estimate, Unsure)
    entry = decision.record.entries["d:calendar/chosen"]
    assert entry["marker"] == {
        "kind": "Unsure",
        "cause": {"code": "required", "refs": ["d:is_calendar"]},
    }


def test_a_port_bound_to_a_field_path_passes_the_field_through() -> None:
    selector = dataclasses.replace(
        SELECTOR,
        rules={
            **SELECTOR.rules,
            "picked": Table(
                "d:picked",
                t.record(calendar=t.boolean),
                "F",
                ["d:is_calendar"],
                [["-", {"calendar": True}]],
            ),
        },
    )
    # Every turn the selector holds calendar true, so email is chosen too.
    design = combine(
        Part(selector),
        EMAIL,
        CALENDAR,
        bind={**BIND, "d:email/chosen": "d:picked.calendar"},
        policy=ACT,
    )
    assert decided(design, "email").record.entries["d:email/chosen"]["value"] is True


def test_a_port_bound_to_a_function_rule_decides() -> None:
    rule = Rule("d:email/chosen", t.boolean, email_chosen, version="1", reads=["d:route"])
    design = combined(bind={**BIND, "d:email/chosen": rule})
    assert design.rules["email/chosen"] is rule
    assert decided(design, "email").estimate == "file"


def test_the_top_policy_may_be_a_function_rule() -> None:
    def act(r: ReadView) -> object:
        return r.read("d:email/outcome")

    policy = Rule("d:act", t.label(*ACTS), act, version="1", reads=["d:email/outcome"])
    assert decided(combined(policy=policy), "email").estimate == "file"


# --- front-loading and the record ----------------------------------------------------------


def test_the_parts_round_one_sources_are_one_call() -> None:
    log = decided(combined(), "email").record.log
    calls = [item for item in log if item["item"] == "call"]
    assert [c["sources"] for c in calls] == [["calendar/ask", "email/ask", "router"]]
    assert calls[0]["round"] == 1


def test_a_combined_record_replays_with_no_findings() -> None:
    loaded = load(combined())
    decision = decide_sync(loaded, {"message": "hello"}, {"default": model("calendar")})
    rerun, findings = replay_sync(decision.record, loaded)
    assert findings == []
    assert rerun is not None
    assert rerun.record_hash == decision.record.record_hash


def test_a_combined_document_reloads_by_import_under_one_hash() -> None:
    loaded = load(combined())
    again = load(loaded.document)
    assert again.design_hash == loaded.design_hash
    assert again.document == loaded.document
    assert load(combined()).design_hash == loaded.design_hash


def test_a_combined_design_with_a_function_port_reloads_by_import() -> None:
    rule = Rule("d:email/chosen", t.boolean, email_chosen, version="1", reads=["d:route"])
    loaded = load(combined(bind={**BIND, "d:email/chosen": rule}))
    assert load(loaded.document).design_hash == loaded.design_hash


def test_the_lint_and_the_read_points_reach_a_combined_design() -> None:
    loaded = load(combined())
    report = dead_questions(loaded, values={"t:state.message": ["hello"]})
    assert set(report["questions"]) == {"q:route", "q:email/sure", "q:calendar/sure"}
    assert {q["status"] for q in report["questions"].values()} == {"live"}
    assert points(loaded, "d:email/chosen") == [False, True]


# --- standalone ----------------------------------------------------------------------------


@pytest.mark.parametrize(("chosen", "outcome"), [(True, "file"), (False, "skip")])
def test_standalone_decides_a_part_alone(chosen: bool, outcome: str) -> None:
    design = standalone(EMAIL, {"d:email/chosen": chosen})
    assert design.policy == "d:email/outcome"
    assert design.name == "email"
    loaded = load(design)
    decision = decide_sync(loaded, {"message": "hi"}, {"default": fixed({"q:email/sure": SURE})})
    assert decision.estimate == outcome
    assert load(loaded.document).design_hash == loaded.design_hash


def test_standalone_takes_unsure_for_a_port() -> None:
    loaded = load(standalone(EMAIL, {"d:email/chosen": Unsure}))
    decision = decide_sync(loaded, {"message": "hi"}, {"default": fixed({"q:email/sure": SURE})})
    assert isinstance(decision.estimate, Unsure)


def test_standalone_refuses_a_missing_or_an_extra_port_value() -> None:
    with pytest.raises(InvalidCombination) as caught:
        standalone(EMAIL, {"d:other": True})
    assert [f.message for f in caught.value.failures] == [
        'the port "d:email/chosen" is given no value',
        'ports names "d:other", which is no port',
    ]


@pytest.mark.parametrize(("chosen", "outcome"), [(True, "file"), (False, "skip")])
def test_a_given_port_takes_its_value_each_turn_from_the_ports_source(
    chosen: bool, outcome: str
) -> None:
    loaded = load(standalone(EMAIL, given=["d:email/chosen"]))
    source = loaded._design.sources["ports"]  # pyright: ignore[reportPrivateUsage]
    assert (source.oracle, list(source.provides)) == ("ports", ["t:email/chosen"])
    bindings = {
        "default": fixed({"q:email/sure": SURE}),
        "ports": fixed({"t:email/chosen": chosen}),
    }
    decision = decide_sync(loaded, {"message": "hi"}, bindings)
    assert decision.estimate == outcome
    _, findings = replay_sync(decision.record, loaded)
    assert findings == []


def test_standalone_refuses_a_port_in_both_or_given_that_is_no_port() -> None:
    with pytest.raises(InvalidCombination) as caught:
        standalone(EMAIL, {"d:email/chosen": True}, given=["d:email/chosen", "d:other"])
    assert [f.message for f in caught.value.failures] == [
        'the port "d:email/chosen" is in both ports and given',
        'given names "d:other", which is no port',
    ]


def test_standalone_refuses_a_ports_source_or_t_ref_the_part_declares() -> None:
    sources = {**EMAIL.design.sources, "ports": Source({"t:email/chosen": t.boolean})}
    part = Part(dataclasses.replace(EMAIL.design, sources=sources), EMAIL.ports)
    with pytest.raises(InvalidCombination) as caught:
        standalone(part, given=["d:email/chosen"])
    assert [f.message for f in caught.value.failures] == [
        'the given port "d:email/chosen" needs "t:email/chosen", which the part provides',
        'the source for the given ports takes the id "ports", which part 1 ("email") declares',
    ]


def test_standalone_refuses_a_port_the_part_provides() -> None:
    part = Part(SELECTOR, {"d:route": t.label(*ROUTES)})
    with pytest.raises(InvalidCombination, match="is provided by part 1"):
        standalone(part, {"d:route": "email"})


# --- refusals ------------------------------------------------------------------------------


def test_parts_whose_states_differ_are_refused() -> None:
    other = Part(dataclasses.replace(EMAIL.design, state=t.record(request=t.string)), EMAIL.ports)
    (message,) = refused(Part(SELECTOR), other, CALENDAR)
    assert message.startswith('the parts\' states differ: part 1 ("selector") declares')
    assert '"record{request: string}"' in message


def _extra(design: Design, **changes: Any) -> Design:
    fields: dict[str, Any] = {
        "types": {"yes_no": YES_NO},
        "groups": {"g": Group(["d:route"], ["convergent"])},
        "targets": {"tg:x": Target("x")},
    }
    fields.update(changes)
    return dataclasses.replace(design, **fields)


def test_a_repeated_id_or_parameter_path_is_refused_even_when_alike() -> None:
    selector = dataclasses.replace(
        SELECTOR,
        sources={**SELECTOR.sources, "email/ask": SELECTOR.sources["router"]},
        parameters={**SELECTOR.parameters, "shared/bar": Parameter(BAR, 0.5)},
    )
    email = Part(
        dataclasses.replace(
            EMAIL.design, parameters={**EMAIL.design.parameters, "shared/bar": Parameter(BAR, 0.5)}
        ),
        EMAIL.ports,
    )
    assert refused(Part(selector), email, CALENDAR) == [
        'the id "email/ask" is declared by part 1 ("selector") and part 2 ("email")',
        'the parameter path "shared/bar" is declared by part 1 ("selector") and part 2 ("email")',
    ]


def test_types_groups_and_targets_declared_alike_are_kept_once() -> None:
    email = Part(_extra(EMAIL.design), EMAIL.ports)
    calendar = Part(_extra(CALENDAR.design), CALENDAR.ports)
    design = combine(Part(_extra(SELECTOR)), email, calendar, bind=BIND, policy=ACT)
    assert dict(design.types) == {"yes_no": YES_NO}
    assert list(design.groups) == ["g"]
    assert list(design.targets) == ["tg:x"]
    load(design)


def test_types_groups_and_targets_declared_differently_are_refused() -> None:
    email = Part(
        _extra(
            EMAIL.design,
            types={"yes_no": t.string},
            groups={"g": Group(["d:route", "d:email/sure"], ["convergent"])},
            targets={"tg:x": Target("y")},
        ),
        EMAIL.ports,
    )
    assert refused(Part(_extra(SELECTOR)), email, CALENDAR) == [
        'part 1 ("selector") and part 2 ("email") declare different groups with the id "g"',
        'part 1 ("selector") and part 2 ("email") declare different targets with the id "tg:x"',
        'part 1 ("selector") and part 2 ("email") declare different types named "yes_no"',
    ]


def test_sources_and_rules_share_one_id_namespace() -> None:
    selector = dataclasses.replace(
        SELECTOR, rules={**SELECTOR.rules, "email/ask": SELECTOR.rules["is_calendar"]}
    )
    assert refused(Part(selector), EMAIL, CALENDAR) == [
        'the id "email/ask" is declared by part 1 ("selector") and part 2 ("email")'
    ]


def test_the_combined_design_holds_the_design_fields_and_no_class_block() -> None:
    design = combine(Part(SELECTOR), EMAIL, CALENDAR, bind=BIND, policy=ACT)
    fields = {f.name for f in dataclasses.fields(design)}
    assert "classes" not in fields
    assert fields == {f.name for f in dataclasses.fields(Design)}


def test_a_port_some_part_provides_is_refused() -> None:
    email = Part(EMAIL.design, {**EMAIL.ports, "d:route": t.label(*ROUTES)})
    assert refused(Part(SELECTOR), email, CALENDAR, bind={**BIND, "d:route": "d:is_calendar"}) == [
        'the port "d:route" is provided by part 1 ("selector"), not a port'
    ]


def test_an_unbound_port_and_a_bind_key_that_is_no_port_are_refused() -> None:
    bind = {"d:email/chosen": EMAIL_CHOSEN, "d:nothing": "d:route"}
    assert refused(Part(SELECTOR), EMAIL, CALENDAR, bind=bind) == [
        'the port "d:calendar/chosen" is not bound',
        'bind names "d:nothing", which is no port',
    ]


def test_a_malformed_port_is_refused() -> None:
    email = Part(EMAIL.design, {"d:email/chosen.x": t.boolean, "q:x": t.boolean, "d:y": "list<"})
    assert refused(Part(SELECTOR), email, bind={}) == [
        'part 2 ("email")\'s port "d:email/chosen.x" is not a d: ref with no field path',
        'part 2 ("email")\'s port "d:y" has type "list<", which is no type',
        'part 2 ("email")\'s port "q:x" is not a d: ref with no field path',
    ]


def test_a_port_declared_with_two_types_is_refused() -> None:
    calendar = Part(CALENDAR.design, {**CALENDAR.ports, "d:email/chosen": t.label("yes")})
    assert refused(Part(SELECTOR), EMAIL, calendar) == [
        'part 2 ("email") and part 3 ("calendar") declare the port "d:email/chosen" with types '
        '"boolean" and "label{\\"yes\\"}"'
    ]


def test_a_provider_for_another_ref_or_of_another_type_is_refused() -> None:
    other = dataclasses.replace(EMAIL_CHOSEN, provides="d:elsewhere")
    wrong = Table("d:email/chosen", t.label("yes"), "F", ["d:route"], [["-", "yes"]])
    assert refused(Part(SELECTOR), EMAIL, CALENDAR, bind={**BIND, "d:email/chosen": other}) == [
        'bind\'s provider for "d:email/chosen" is a rule providing "d:elsewhere"'
    ]
    assert refused(Part(SELECTOR), EMAIL, CALENDAR, bind={**BIND, "d:email/chosen": wrong}) == [
        'bind\'s provider for "d:email/chosen" outputs "label{\\"yes\\"}", which does not '
        'conform to the port\'s "boolean"'
    ]


def test_a_ref_provider_that_is_no_d_ref_is_refused() -> None:
    assert refused(Part(SELECTOR), EMAIL, CALENDAR, bind={**BIND, "d:email/chosen": "q:route"}) == [
        'bind\'s provider for "d:email/chosen" is "q:route", which is no d: ref'
    ]


def test_an_added_rule_whose_id_a_part_declares_is_refused() -> None:
    policy = dataclasses.replace(ACT, provides="d:route")
    assert refused(Part(SELECTOR), EMAIL, CALENDAR, policy=policy) == [
        'the rule for "d:route" takes the id "route", which part 1 ("selector") declares'
    ]


def test_every_failure_is_listed_at_once() -> None:
    other = Part(dataclasses.replace(EMAIL.design, state=t.record()), EMAIL.ports)
    messages = refused(Part(SELECTOR), other, Part(SELECTOR), bind={})
    assert len(messages) > 3
    assert any(m.startswith("the parts' states differ") for m in messages)
    assert 'the port "d:email/chosen" is not bound' in messages


def test_load_still_checks_what_combine_leaves_to_it() -> None:
    from ommat import InvalidDesign

    bind = {**BIND, "d:calendar/chosen": "d:nowhere"}
    with pytest.raises(InvalidDesign, match="d:nowhere"):
        load(combined(bind=bind))


def test_a_failure_names_what_it_concerns_in_code_point_order() -> None:
    with pytest.raises(InvalidCombination) as caught:
        combine(Part(SELECTOR), EMAIL, CALENDAR, bind={}, policy=ACT)
    assert caught.value.failures == (
        CombineFailure(("d:calendar/chosen",), 'the port "d:calendar/chosen" is not bound'),
        CombineFailure(("d:email/chosen",), 'the port "d:email/chosen" is not bound'),
    )
    assert str(caught.value) == "\n".join(f.message for f in caught.value.failures)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: combine(bind={}, policy=ACT), "at least one part"),
        (lambda: combine(SELECTOR, bind={}, policy=ACT), "not a Part"),  # type: ignore[arg-type]
        (lambda: combine(Part(SELECTOR), bind=[], policy=ACT), "bind is a list"),  # type: ignore[arg-type]
        (lambda: combine(Part(SELECTOR), bind={}, policy="d:route"), "not a Rule or a Table"),  # type: ignore[arg-type]
        (lambda: combine(EMAIL, bind={"d:email/chosen": 1}, policy=ACT), "not a d: ref"),  # type: ignore[dict-item]
        (lambda: standalone(EMAIL, []), "ports are a list"),  # type: ignore[arg-type]
    ],
)
def test_a_wrong_argument_type_is_a_type_error(call: Any, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        call()
