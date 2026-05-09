"""Sphinx directive for embedding the Rerun web viewer.

Use this from any tutorial or page to drop in an interactive Rerun viewer
backed by an ``.rrd`` recording shipped under ``_static/``::

    .. rerun-embed:: nuscenes_mini.rrd
       :height: 700px
       :width: 100%

The directive emits a single ``<div class="rerun-embed">`` carrying the
recording path and SDK version as ``data-*`` attributes. The shared
``_static/rerun_embed.js`` loader (registered by :func:`setup`) finds every
such element on the page and instantiates ``@rerun-io/web-viewer`` against
it.
"""

from collections.abc import Callable
from typing import ClassVar, override

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata


class RerunEmbedDirective(Directive):
    """Embed an interactive Rerun web viewer.

    Args:
        rrd: Basename of the ``.rrd`` file under ``docs/source/_static/``.

    Options:
        height: CSS height for the viewer container. Defaults to ``600px``.
        width: CSS width for the viewer container. Defaults to ``100%``.
    """

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False
    option_spec: ClassVar[dict[str, Callable[[str], str]]] = {
        "height": directives.unchanged,
        "width": directives.unchanged,
    }

    @override
    def run(self) -> list[nodes.Node]:
        """Render the embed as a raw-HTML node.

        Returns:
            A single ``nodes.raw`` HTML node — a ``<div>`` carrying the rrd
            path and SDK version on ``data-*`` attributes for the shared
            loader to pick up.
        """
        import rerun

        rrd_name = self.arguments[0]
        height = self.options.get("height", "600px")
        width = self.options.get("width", "100%")

        # Compute the relative path from this page to ``_static/`` so the
        # embed works regardless of deploy subpath.
        docname = self.state.document.settings.env.docname
        depth = docname.count("/")
        rel_to_static = "../" * depth + "_static/" + rrd_name

        html = (
            f'<div class="rerun-embed"'
            f' data-rrd="{rel_to_static}"'
            f' data-rerun-version="{rerun.__version__}"'
            f' style="width:{width};height:{height};"></div>'
        )
        return [nodes.raw("", html, format="html")]


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register :class:`RerunEmbedDirective` with Sphinx.

    Args:
        app: The Sphinx application to register against.

    Returns:
        Extension metadata advertising version and parallel-safety.
    """
    app.add_directive("rerun-embed", RerunEmbedDirective)
    app.add_js_file("rerun_embed.js", type="module")
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
