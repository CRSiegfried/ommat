import importlib
import subprocess
import sys
from importlib.metadata import version

import pytest

import ommat


def test_version_matches_metadata() -> None:
    assert ommat.__version__ == version("ommat")


@pytest.mark.parametrize("module", ["ommat", "ommat.conversation", "ommat.lens", "ommat.outline"])
def test_core_imports_only_the_standard_library(module: str) -> None:
    # A fresh interpreter, so modules loaded by pytest or plugins don't count.
    probe = (
        f"import sys; before = set(sys.modules); import {module}; "
        "names = {m.partition('.')[0] for m in set(sys.modules) - before}; "
        "print('\\n'.join(sorted(names - set(sys.stdlib_module_names) - {'ommat'})))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == []


# The public API names.
PUBLIC = {
    "transcript",
    "Design",
    "DesignFailure",
    "Group",
    "Helper",
    "Input",
    "InvalidDesign",
    "LoadedDesign",
    "Param",
    "Parameter",
    "Question",
    "Rule",
    "Source",
    "SourceRequirement",
    "When",
    "Table",
    "Target",
    "TargetRole",
    "Structured",
    "Text",
    "load",
    "Answer",
    "AnswerKind",
    "ExtractionAnswer",
    "ValueType",
    "Cause",
    "CauseCode",
    "Failed",
    "InvalidRecord",
    "Record",
    "Unasked",
    "Unavailable",
    "Undetermined",
    "Unsure",
    "read_record",
    "CallResult",
    "SourceCall",
    "RenderedQuestion",
    "ReadView",
    "RunSettings",
    "Decision",
    "DefectWarning",
    "EntryRefusal",
    "decide",
    "decide_sync",
    "defects",
    "escalation",
    "redact",
    "replay",
    "replay_sync",
    "FindingKind",
    "from_record",
}
TYPE_CONSTRUCTORS = {
    "number",
    "integer",
    "string",
    "boolean",
    "bytes_",
    "label",
    "list_",
    "mapping",
    "record",
    "nullable",
}


def test_the_public_names_are_exactly_the_decisions() -> None:
    assert set(ommat.__all__) == {*PUBLIC, "__version__"}
    public = {n for n in vars(ommat) if not n.startswith("_")}
    # Submodules imported by tests appear as attributes; `types`, `helpers`, `evaluation` and
    # `bindings` are public subpackages, not names in `ommat.__all__`.
    assert (
        public - {"types", "helpers", "evaluation", "bindings", "conversation", "lens", "outline"}
        == PUBLIC
    )


BINDINGS = {
    "OracleBinding",
    "RequestProvenance",
    "bounded_warnings",
    "call_result",
    "candidate_of",
    "extraction_value",
    "fixed",
    "image_need",
    "json_value",
    "prompt_text",
}


def test_ommat_bindings_exports_exactly_the_binding_kit() -> None:
    import ommat.bindings

    assert "bindings" not in ommat.__all__
    assert set(ommat.bindings.__all__) == BINDINGS
    public = {n for n in vars(ommat.bindings) if not n.startswith("_")}
    # The suite and the shipped bindings are submodules, attributes only once a test imports them.
    assert public - {"conformance", "idia"} == BINDINGS


def test_import_ommat_imports_no_binding_module() -> None:
    probe = (
        "import sys, ommat; print(sorted(m for m in sys.modules if m.startswith('ommat.bindings')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_import_ommat_imports_no_conversation_module() -> None:
    probe = (
        "import sys, ommat; "
        "print(sorted(m for m in sys.modules if m.startswith('ommat.conversation')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_import_ommat_imports_no_outline_module() -> None:
    probe = (
        "import sys, ommat; print(sorted(m for m in sys.modules if m.startswith('ommat.outline')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_ommat_outline_exports_exactly_its_builders_and_renderers() -> None:
    import ommat.outline

    assert "outline" not in ommat.__all__
    assert ommat.outline.__all__ == [
        "design_outline",
        "render_design",
        "render_turn",
        "turn_outline",
    ]


def test_ommat_conversation_exports_exactly_its_renderers() -> None:
    import ommat.conversation

    assert "conversation" not in ommat.__all__
    assert ommat.conversation.__all__ == [
        "drop_earlier_bookkeeping",
        "is_bookkeeping",
        "to_chat_completions",
    ]


def test_ommat_conversation_pydantic_ai_exports_exactly_its_renderer() -> None:
    pytest.importorskip("pydantic_ai")
    import ommat.conversation.pydantic_ai

    assert ommat.conversation.pydantic_ai.__all__ == [
        "drop_earlier_bookkeeping",
        "is_bookkeeping",
        "to_model_messages",
    ]


def test_the_pydantic_ai_extra_names_its_floor_and_the_conversation_extra_is_gone() -> None:
    from importlib.metadata import metadata

    requires = metadata("ommat").get_all("Requires-Dist") or []
    assert "pydantic-ai-slim>=2.49; extra == 'pydantic-ai'" in requires
    assert "conversation" not in (metadata("ommat").get_all("Provides-Extra") or [])


EVALUATION = {
    "DesignStore",
    "EvaluationRefusal",
    "EvaluationStore",
    "RefusalReason",
    "answers",
    "deciding_inputs",
    "deciding_inputs_sync",
    "gold_record",
    "gold_record_sync",
    "intervene",
    "intervene_sync",
    "question_reach",
    "question_reach_sync",
    "rescore",
    "rescore_sync",
    "scenario",
    "scenario_sync",
    "switching_values",
    "switching_values_batch",
    "switching_values_batch_sync",
    "switching_values_sync",
    "use_answers",
    "use_answers_sync",
    "write_gold_answers",
}


CUT = {
    f"{name}{twin}"
    for name in ("set_entry", "fail_call", "set_state_field", "constant_rule")
    for twin in ("", "_sync")
}
"""The single-intervention jobs, whose forms are ``intervene``'s intervention kinds."""


def test_ommat_evaluation_exports_exactly_its_jobs_stores_and_refusal() -> None:
    import ommat.evaluation

    assert not CUT & set(dir(ommat.evaluation))

    assert "evaluation" not in ommat.__all__
    assert set(ommat.evaluation.__all__) == EVALUATION
    assert {n for n in vars(ommat.evaluation) if not n.startswith("_")} == EVALUATION


def test_ommat_types_exports_exactly_the_constructors() -> None:
    import ommat.types

    assert set(ommat.types.__all__) == TYPE_CONSTRUCTORS
    assert {n for n in vars(ommat.types) if not n.startswith("_")} == TYPE_CONSTRUCTORS


def test_plain_literals_and_lambdas_build_the_public_types() -> None:
    import typing_strict

    assert typing_strict.source.needs == ("t:state",)
    assert typing_strict.design.policy == "d:act"


GONE = {
    "Binding",
    "Context",
    "ContextEntry",
    "GuardFailure",
    "Origin",
    "Plan",
    "PlanDeclaration",
    "PlanKind",
    "Policy",
    "PolicyTable",
    "Requirement",
    "Step",
    "Verdict",
    "VerdictKind",
    "handoff",
}
"""Names no public module holds."""


def test_no_public_module_holds_a_plan_guard_or_context_name() -> None:
    import ommat.conversation
    import ommat.helpers

    assert not GONE & {n for n in vars(ommat) if not n.startswith("_")}
    assert "handoff_text" not in vars(ommat.conversation)
    assert not {"BindingNotFound", "RunRefusal", "run_plan", "unsure_to_execute"} & set(
        vars(ommat.helpers)
    )


@pytest.mark.parametrize(
    "name", ["ommat", "ommat.types", "ommat.helpers", "ommat.evaluation", "ommat.bindings"]
)
def test_each_public_module_publishes_its_all(name: str) -> None:
    module = importlib.import_module(name)
    public: list[str] = list(getattr(module, "__all__", []))
    assert public, f"{name} defines no __all__"
    assert len(public) == len(set(public)), f"{name}.__all__ repeats a name"
    for item in public:
        dunder = item.startswith("__") and item.endswith("__")
        assert dunder or not item.startswith("_"), f"{name}.__all__ lists {item}"
        assert hasattr(module, item), f"{name}.__all__ lists {item}, which it does not hold"


def test_ci_installs_the_pydantic_ai_extra_so_its_tests_never_skip_there() -> None:
    import importlib.util
    import os

    if not os.environ.get("CI"):
        pytest.skip("only CI must hold every extra")
    assert importlib.util.find_spec("pydantic_ai") is not None
