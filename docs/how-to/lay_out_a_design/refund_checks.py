"""Two checks a design module's tests can hold: that its docstring map names the refs the design
provides, and that a tidy which renamed or moved rule functions changed nothing else."""

import json
import re
from types import ModuleType
from typing import Any

from ommat import Design, load
from ommat.outline import design_outline

MAP_REF = re.compile(r"``([qtd]:[\w/]+)``")


def map_disagrees(module: ModuleType, design: Design) -> dict[str, list[str]]:
    """The refs the module's docstring names that the design does not provide, and those it
    provides that the docstring leaves out; both empty when the map is true."""
    named = set(MAP_REF.findall(module.__doc__ or ""))
    outline = design_outline(load(design).document)
    provided = {ref for node in outline["nodes"] if node["kind"] != "state" for ref in node["refs"]}
    return {"not provided": sorted(named - provided), "not in the map": sorted(provided - named)}


def moved_nothing(before: Design, after: Design) -> bool:
    """Whether two designs differ only in where their rule functions live and what they are
    called: each function rule's module, qualname and source dropped, the documents are equal."""
    return _without_places(before) == _without_places(after)


def _without_places(design: Design) -> dict[str, Any]:
    document = json.loads(load(design).document)
    for rule in document["rules"].values():
        if rule.get("form") == "function":
            for member in ("module", "qualname", "source"):
                rule.pop(member, None)
    return document
