"""Blueprint helpers for arranging Rerun views from vision3d datasets."""

from collections.abc import Sequence

from ._rerun import rr, rrb


def camera_grid(
    camera_names: Sequence[str],
    grid: Sequence[Sequence[int]] | None = None,
    *,
    entity_prefix: str = "world/cam",
    overlay_entities: Sequence[str] | None = ("world/gt/boxes", "world/pred/boxes"),
) -> rrb.Grid:
    """Build a 2D camera-panel grid from a dataset's rig metadata.

    Each cell in ``grid`` is an index into ``camera_names``. Entity origins
    follow ``log_cameras``' ``{entity_prefix}/{i}`` convention so this helper
    pairs directly with :func:`vision3d.viz.log_cameras`.

    Panels are emitted row-major into a :class:`~rerun.blueprint.Grid`
    with ``grid_columns`` set to the widest row.

    Args:
        camera_names: Per-camera display names indexed by tensor position.
        grid: Row-major grid of indices into ``camera_names``. ``None`` if
            the dataset hasn't declared a rig layout. Falls back to a single
            row in tensor order.
        entity_prefix: Prefix for camera entity origins (e.g. ``"world/cam"``
            -> ``/world/cam/0``, ``/world/cam/1`` ...).
        overlay_entities: Box entities to overlay on every camera panel
            (e.g. ``("world/gt/boxes", "world/pred/boxes")``). All overlays
            are rendered as ``"majorwireframe"`` in the projections, since
            filled boxes would occlude the underlying image. Pass ``None`` or
            an empty sequence to skip the overlay.

    Returns:
        A :class:`~rerun.blueprint.Grid` containing one
        :class:`~rerun.blueprint.Spatial2DView` per declared camera.

    Raises:
        ValueError: If any index is out of range for ``camera_names``.
    """
    if grid is None:
        grid = (tuple(range(len(camera_names))),)

    cols = max(len(row) for row in grid)
    overlays = list(overlay_entities or ())

    # Box overlays are rendered as wireframes so filled faces don't occlude
    # the image. Contents are the same for every panel.
    contents = ["+ $origin/**", *(f"+ /{entity}/**" for entity in overlays)]

    panels = []
    for row in grid:
        for idx in row:
            if not 0 <= idx < len(camera_names):
                msg = f"grid index {idx} out of range for {len(camera_names)} cameras"
                raise ValueError(msg)
            panels.append(
                rrb.Spatial2DView(
                    name=camera_names[idx],
                    origin=f"/{entity_prefix}/{idx}",
                    contents=contents,
                    overrides={
                        f"/{entity}": rr.Boxes3D.from_fields(fill_mode="majorwireframe")
                        for entity in overlays
                    }
                    or None,
                )
            )
    return rrb.Grid(*panels, grid_columns=cols)


def lidar_view(
    *,
    entity_prefix: str = "world",
    name: str = "3D",
) -> rrb.Spatial3DView:
    """Build a 3D view of the world entity tree.

    The view captures everything under ``/{entity_prefix}``, typically the
    lidar point cloud, 3D boxes, and any logged camera frustums. Pairs with
    :func:`vision3d.viz.log_point_cloud` and :func:`vision3d.viz.log_sample`.

    Args:
        entity_prefix: Origin entity path (without leading slash).
        name: Display name shown in the view's title bar.

    Returns:
        A :class:`~rerun.blueprint.Spatial3DView` rooted at
        ``/{entity_prefix}``.
    """
    return rrb.Spatial3DView(origin=f"/{entity_prefix}", name=name)


def time_series_view(
    *,
    entity_prefix: str = "train",
    name: str = "metrics",
) -> rrb.TimeSeriesView:
    """Build a time-series view of logged training metrics.

    Captures every scalar logged under ``/{entity_prefix}``, pairing with
    :func:`vision3d.viz.log_scalars` to plot quantities such as loss and
    learning rate over a training run.

    Args:
        entity_prefix: Origin entity path (without leading slash), matching
            :func:`vision3d.viz.log_scalars`' ``prefix``.
        name: Display name shown in the view's title bar.

    Returns:
        A :class:`~rerun.blueprint.TimeSeriesView` rooted at
        ``/{entity_prefix}``.
    """
    return rrb.TimeSeriesView(origin=f"/{entity_prefix}", name=name)


def fusion_layout(
    camera_names: Sequence[str],
    grid: Sequence[Sequence[int]] | None = None,
    *,
    entity_prefix: str = "world",
    row_shares: Sequence[int] = (3, 2),
    name: str | None = None,
) -> rrb.Vertical:
    """Build a fusion-sample layout with a 3D view above a camera grid.

    Composes :func:`lidar_view` and :func:`camera_grid` under matching entity
    prefixes that align with :func:`vision3d.viz.log_sample`'s defaults
    (``world/cam/*`` for cameras, ``world/gt/boxes`` and ``world/pred/boxes``
    for the box overlays).

    Args:
        camera_names: Per-camera display names indexed by tensor position.
        grid: Row-major grid of indices into ``camera_names``. See
            :func:`camera_grid`.
        entity_prefix: Root entity prefix; the 3D view roots at
            ``/{entity_prefix}``, cameras at ``/{entity_prefix}/cam/*``,
            box overlays at ``/{entity_prefix}/gt/boxes`` and
            ``/{entity_prefix}/pred/boxes``.
        row_shares: Vertical split ratio between the 3D view and camera grid.
        name: Optional display name.

    Returns:
        A :class:`~rerun.blueprint.Vertical` container stacking the 3D view
        and camera grid.
    """
    return rrb.Vertical(
        lidar_view(entity_prefix=entity_prefix),
        camera_grid(
            camera_names,
            grid,
            entity_prefix=f"{entity_prefix}/cam",
            overlay_entities=(
                f"{entity_prefix}/gt/boxes",
                f"{entity_prefix}/pred/boxes",
            ),
        ),
        row_shares=list(row_shares),
        name=name,
    )
