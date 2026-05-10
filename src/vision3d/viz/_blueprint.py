"""Blueprint helpers for arranging Rerun views from vision3d datasets."""

from collections.abc import Sequence

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError as e:
    msg = "rerun-sdk is required for visualization. Install with: pip install vision3d[viz]"
    raise ImportError(msg) from e


def camera_grid(
    camera_names: Sequence[str],
    grid: Sequence[Sequence[int]] | None = None,
    *,
    entity_prefix: str = "world/cam",
    overlay_entity: str | None = "world/boxes",
) -> rrb.Grid:
    """Build a 2D camera-panel grid from a dataset's rig metadata.

    Each cell in ``grid`` is an index into ``camera_names``. Entity origins
    follow ``log_cameras``' ``{entity_prefix}_{i}`` convention so this helper
    pairs directly with :func:`vision3d.viz.log_cameras`.

    ``grid`` rows may be ragged. The grid is flattened row-major and wrapped
    at the widest row's column count.

    Args:
        camera_names: Per-camera display names indexed by tensor position.
        grid: Row-major grid of indices into ``camera_names``. ``None`` if
            the dataset hasn't declared a rig layout. Falls back to a single
            row in tensor order.
        entity_prefix: Prefix for camera entity origins (e.g. ``"world/cam"``
            -> ``/world/cam_0``, ``/world/cam_1`` ...).
        overlay_entity: Entity to overlay as wireframe boxes on every camera
            panel (e.g. ``"world/boxes"``). Pass ``None`` to skip the overlay.

    Returns:
        An ``rrb.Grid`` containing one ``rrb.Spatial2DView`` per declared
        camera.

    Raises:
        ValueError: If any index is out of range for ``camera_names``.
    """
    if grid is None:
        grid = (tuple(range(len(camera_names))),)

    cols = max(len(row) for row in grid)

    panels = []
    for row in grid:
        for idx in row:
            if not 0 <= idx < len(camera_names):
                msg = f"grid index {idx} out of range for {len(camera_names)} cameras"
                raise ValueError(msg)
            contents = ["+ $origin/**"]
            if overlay_entity is not None:
                contents.append(f"+ /{overlay_entity}/**")
            panels.append(
                rrb.Spatial2DView(
                    name=camera_names[idx],
                    origin=f"/{entity_prefix}_{idx}",
                    contents=contents,
                    overrides=(
                        {
                            f"/{overlay_entity}": rr.Boxes3D.from_fields(
                                fill_mode="majorwireframe"
                            )
                        }
                        if overlay_entity is not None
                        else None
                    ),
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
        An ``rrb.Spatial3DView`` rooted at ``/{entity_prefix}``.
    """
    return rrb.Spatial3DView(origin=f"/{entity_prefix}", name=name)


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
    (``world/cam_*`` for cameras, ``world/boxes`` for the overlay).

    Args:
        camera_names: Per-camera display names indexed by tensor position.
        grid: Row-major grid of indices into ``camera_names``. See
            :func:`camera_grid`.
        entity_prefix: Root entity prefix; the 3D view roots at
            ``/{entity_prefix}``, cameras at ``/{entity_prefix}/cam_*``,
            box overlay at ``/{entity_prefix}/boxes``.
        row_shares: Vertical split ratio between the 3D view and camera grid.
        name: Optional display name.

    Returns:
        An ``rrb.Vertical`` container stacking the 3D view and camera grid.
    """
    return rrb.Vertical(
        lidar_view(entity_prefix=entity_prefix),
        camera_grid(
            camera_names,
            grid,
            entity_prefix=f"{entity_prefix}/cam",
            overlay_entity=f"{entity_prefix}/boxes",
        ),
        row_shares=list(row_shares),
        name=name,
    )
