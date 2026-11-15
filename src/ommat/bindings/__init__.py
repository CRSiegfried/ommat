"""The binding kit, what a binding author calls, and the bindings ommat ships.

``extraction_value`` writes an extraction answer in the form a binding returns, its dates,
times, datetimes and durations in ommat's grammar. ``bounded_warnings`` makes any lines a
conforming ``warnings``. ``json_value`` writes a need's value as JSON in its record form, and
``image_need`` reads an image need's media type and bytes. ``candidate_of`` reads a grounded
ref's candidate, so a binding of a grounded source answers each candidate's ref.
``prompt_text`` writes a prompt as the record's text, and ``call_result`` folds each request's
``RequestProvenance`` into the call's ``CallResult``. ``DETAILS`` names every detail of the
provenance profile, a ``missing`` key or a ``claimed`` name, ``REASONS`` every reason a
``missing`` gives, and ``USAGE_KEYS`` the keys a ``usage`` holds; whatever else a provider reports
goes in a request's ``extras``. ``fixed(values)`` is a binding to no
provider, answering each asked ref from a mapping, for a fact the caller holds or a test's
scripted answers. The binding conformance suite is ``ommat.bindings.conformance``. Each binding
ommat ships to a provider library is a module here behind its own extra, which this package
never imports. The names in ``__all__`` are public under the same deprecation rule as
``ommat.__all__``; the suite and the shipped bindings are outside it. ``import ommat`` imports
none of this package.
"""

from ommat._binding import OracleBinding
from ommat._binding_kit import (
    RequestProvenance,
    bounded_warnings,
    call_result,
    candidate_of,
    extraction_value,
    fixed,
    image_need,
    json_value,
    prompt_text,
)
from ommat._provenance import DETAILS, REASONS, USAGE_KEYS

__all__ = [
    "DETAILS",
    "REASONS",
    "USAGE_KEYS",
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
]
