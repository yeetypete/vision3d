# Instructions for AI agents

vision3d is a 3D extension of [torchvision](https://github.com/pytorch/vision)
providing datasets, tensor types, transforms, ops, metrics, and visualization
utilities for 3D perception.

## Follow other guidelines

Read the vision3d [README](README.md). Follow the
[CONTRIBUTING.md](CONTRIBUTING.md) guidelines for contribution.

## Commits

Use [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/)
prefixes (e.g. `feat:`, `fix:`) with a short imperative summary.

## Attribution

When generating substantial amounts of code, you SHOULD include an
`Assisted-by: TOOLNAME (MODELNAME)` trailer on the commit. For example,
`Assisted-by: Claude Code (Opus 4.7)`.

## Code conventions

- Follow the patterns established by
  [torchvision](https://github.com/pytorch/vision) (API design, module layout,
  naming).
- Transforms and ops must not make assumptions about the scene (e.g. symmetrical
  sensor rigs, fixed camera count, specific axis conventions). Keep them general
  so they work across datasets.
- All transforms must support the full 9-DoF bounding box parameterization (xyz
  center + xyz size + roll/pitch/yaw), not just the 7-DoF (yaw-only) subset.
- Write Google-style docstrings, parsed by Sphinx via the
  [Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
  extension and enforced by ruff's `D`/`DOC` rules. Document constructor args in
  the class docstring rather than `__init__` (`D107` is ignored).
- Private modules use a leading underscore (e.g. `_box3d_iou.py`). Public APIs
  are re-exported from each submodule's `__init__.py` without the underscore.
