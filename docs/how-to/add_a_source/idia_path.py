"""Path A: the design's oracle answered by an idia provider, through ``ommat[idia]``."""

import idia

from ommat.bindings import OracleBinding
from ommat.bindings.idia import oracle_binding


def idia_bindings(provider: idia.Dispatcher) -> dict[str, OracleBinding]:
    """The design's one oracle, answered by an idia provider."""
    return {"spans": oracle_binding(provider)}
