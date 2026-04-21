"""Render GitHub-flavored Markdown alert blockquotes as Sphinx admonitions.

MyST-parser does not yet natively support the ``> [!NOTE]`` / ``[!TIP]`` /
``[!IMPORTANT]`` / ``[!WARNING]`` / ``[!CAUTION]`` syntax that GitHub renders
as styled callouts (see `executablebooks/MyST-Parser#845`_). This extension adds
a docutils transform that rewrites those blockquotes into the corresponding
admonition nodes so they render consistently on GitHub and in the Sphinx build.

.. _executablebooks/MyST-Parser#845:
   https://github.com/executablebooks/MyST-Parser/issues/845
"""

import re

from docutils import nodes
from docutils.transforms import Transform
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

# Match GitHub's strictness: the marker must be on its own line (only horizontal
# whitespace permitted around it), and content follows on subsequent lines.
_ALERT_RE = re.compile(r"^[ \t]*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*$")

_ALERT_TO_NODE: dict[str, type[nodes.Element]] = {
    "NOTE": nodes.note,
    "TIP": nodes.tip,
    "IMPORTANT": nodes.important,
    "WARNING": nodes.warning,
    "CAUTION": nodes.danger,
}


class GfmAlertTransform(Transform):
    """Rewrite GFM-alert blockquotes as admonitions."""

    default_priority = 500

    def apply(self, **kwargs: object) -> None:
        """Walk the document and convert matching block quotes in place."""
        for block_quote in list(self.document.findall(nodes.block_quote)):
            first = block_quote.children[0] if block_quote.children else None
            if not isinstance(first, nodes.paragraph) or not first.children:
                continue
            head = first.children[0]
            if not isinstance(head, nodes.Text):
                continue
            match = _ALERT_RE.match(head.astext())
            if match is None:
                continue

            remainder = head.astext()[match.end() :].lstrip("\n")
            if remainder:
                first.replace(head, nodes.Text(remainder))
            else:
                first.remove(head)
                if not first.children:
                    block_quote.remove(first)

            admonition = _ALERT_TO_NODE[match.group(1)]()
            admonition.extend(block_quote.children)
            block_quote.replace_self(admonition)


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register :class:`GfmAlertTransform` with Sphinx.

    Args:
        app: The Sphinx application to register against.

    Returns:
        Extension metadata advertising version and parallel-safety.
    """
    app.add_transform(GfmAlertTransform)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
