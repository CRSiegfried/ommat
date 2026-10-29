# pyright: reportPrivateUsage=false
"""The jobs given ``loaded`` that load an edited design, ``constant_rule``, ``intervene``,
``deciding_inputs`` and ``switching_values``, take ``loaded``'s own callables when
``functions`` is ``None``, so a design of lambdas and closures needs none passed; explicit
``functions`` still override them (§evaluation-jobs)."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ommat import (
    CallResult,
    Design,
    InvalidDesign,
    LoadedDesign,
    Parameter,
    Plan,
    Policy,
    Rule,
    Source,
    When,
    decide_sync,
    load,
)
from ommat import types as t
from ommat.evaluation import (
    EvaluationRefusal,
    EvaluationStore,
    RefusalReason,
    constant_rule_sync,
    deciding_inputs_sync,
    intervene_sync,
    switching_values_sync,
)


def closures(prefix: str = "") -> LoadedDesign:
    """A design whose rules, source ``when`` and policy are a lambda and closures, which load
    of its document alone cannot import; ``prefix`` opens the policy's reply, a value its
    closure holds and the document does not."""
    shown = {True: "big", False: "small"}

    def say(r: Any) -> str:
        return shown[bool(r.read("d:big"))]

    def policy(r: Any) -> Plan:
        return Plan("inform", reply=prefix + str(r.read("d:say")))

    return load(
        Design(
            state=t.record(n=t.integer()),
            sources={
                "look": Source(
                    {"t:seen": t.string},
                    needs=["t:state"],
                    when=When(lambda r: r.read("t:state.n") >= 0, version="1"),
                )
            },
            rules={
                "big": Rule(
                    "d:big",
                    t.boolean,
                    lambda r: r.read("t:state.n") > r.param("lim/k"),
                    version="1",
                    reads=["t:state.n"],
                    params=["lim/k"],
                    boundaries={"t:state.n": ["lim/k"]},
                ),
                "say": Rule("d:say", t.label("big", "small"), say, version="1", reads=["d:big"]),
            },
            parameters={"lim/k": Parameter(t.integer(0, 10), 3)},
            policy=Policy(policy, version="1", reads=["d:say"]),
        )
    )


LOADED = closures()


async def seen(batch: Any, *, seconds: float) -> CallResult:
    return CallResult({"t:seen": "ok"})


RECORD = decide_sync(LOADED, {"n": 5}, {"default": seen}).record


def test_the_document_alone_does_not_load() -> None:
    with pytest.raises(InvalidDesign):
        load(LOADED.document)


def test_constant_rule_takes_loadeds_own_functions() -> None:
    out = constant_rule_sync(RECORD, LOADED, "d:big", False)
    assert any(f["kind"] == "plan" for f in out["findings"])


def test_intervene_takes_loadeds_own_functions() -> None:
    out = intervene_sync(RECORD, LOADED, [], rule={"rule": "d:say", "output": "small"})
    assert out["document"]["rules"]["say"]["form"] == "table"


def test_deciding_inputs_takes_loadeds_own_functions() -> None:
    assert deciding_inputs_sync(RECORD, LOADED, "d:say") == {"d:big": "deciding"}


def test_switching_values_takes_loadeds_own_functions(tmp_path: Path) -> None:
    entry = switching_values_sync(RECORD, LOADED, "lim/k", results=EvaluationStore(tmp_path))
    plan = entry["value"]["plan"]
    assert [c["least"] for c in plan["classes"]] == [0, 5]  # n > k switches at k = 5


def test_explicit_functions_override_loadeds_own() -> None:
    other = closures("other: ")
    assert other.design_hash == LOADED.design_hash  # the same code, closing over another value
    out = constant_rule_sync(RECORD, LOADED, "d:big", False, functions=other._callables)
    (plan,) = [f for f in out["findings"] if f["kind"] == "plan"]
    assert plan["replayed"]["reply"] == "other: small"


def test_explicit_empty_functions_do_not_fall_back(tmp_path: Path) -> None:
    jobs: list[Callable[[], object]] = [
        lambda: constant_rule_sync(RECORD, LOADED, "d:big", True, functions={}),
        lambda: intervene_sync(
            RECORD, LOADED, [], rule={"rule": "d:big", "output": True}, functions={}
        ),
        lambda: deciding_inputs_sync(RECORD, LOADED, "d:say", functions={}),
        lambda: switching_values_sync(
            RECORD, LOADED, "lim/k", functions={}, results=EvaluationStore(tmp_path)
        ),
    ]
    for job in jobs:
        with pytest.raises(EvaluationRefusal) as raised:
            job()
        assert raised.value.reason is RefusalReason.invalid_design
