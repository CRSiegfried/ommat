"""Read an authoring document, a parsed mapping, as the `Design` or `Slice` it says.

A document is trusted as source code: the reader imports each rule's `fn`, which runs that
module's top-level code. Read only documents you would import as modules.

`design_from(doc, version=None, values=None, types=None)` gives the whole `Design`;
`slice_from(...)` gives the `Slice` of sources, rules and parameters a larger design merges;
`AuthoringError`, a `ValueError`, names the path of a slip in the document. The conventions the
document adds are in the spec, in the section "The authoring reader". This package takes nothing
beyond the standard library and reads no file: parse TOML with `tomllib`, then pass the mapping.
`import ommat` does not import it.
"""

from ommat.authoring._reader import AuthoringError, Slice, design_from, slice_from

__all__ = ["AuthoringError", "Slice", "design_from", "slice_from"]
