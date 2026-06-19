"""
Logging training metrics with vision3d
======================================

This example demonstrates tracking a 3D detector's training run with
`Rerun <https://rerun.io/>`_: per-step scalar metrics
(:func:`vision3d.viz.log_scalars`), comparing several runs in one plot
(:func:`vision3d.viz.style_series`), and watching predictions converge on a
fixed sample over training (:func:`vision3d.viz.log_boxes_3d` with
``static=True``). Everything is logged to a single recording and arranged into
one dashboard with :func:`vision3d.viz.time_series_view` and
:func:`vision3d.viz.lidar_view`. Because every panel shares the ``step``
timeline, playing it draws the loss curves *and* converges the 3D boxes at the
same time.

Training itself lives outside vision3d, but the logging primitives do not. To
keep the example self-contained we synthesize a plausible run rather than
training a real model; in practice the scalars come from your loop and the
validation metrics from :mod:`vision3d.metrics`.
"""

# %%
# Synthesize a training run
# -------------------------
# A real loop would compute these from a model and a dataset. Here we fake a
# BEVFusion-style run: a total loss that decays with noise, split into
# classification, regression, and heatmap components, plus a learning rate
# following linear warmup then cosine decay. ``simulate_run`` is parameterized
# by hyperparameters so we can replay it for several runs later.

import math
from typing import NamedTuple

import torch

EPOCHS = 6
STEPS_PER_EPOCH = 50
WARMUP_STEPS = 30
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH


def lr_at(step: int, base_lr: float) -> float:
    """Linear warmup then cosine decay to zero.

    Returns:
        The learning rate for the given global ``step``.
    """
    if step < WARMUP_STEPS:
        return base_lr * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def simulate_run(
    *, base_lr: float, decay: float, seed: int
) -> list[tuple[dict[str, torch.Tensor], float]]:
    """Synthesize per-step losses and learning rates for one run.

    Returns:
        One ``(losses, lr)`` pair per step, where ``losses`` maps component
        name to a scalar-tensor loss.
    """
    generator = torch.Generator().manual_seed(seed)
    steps = []
    for step in range(TOTAL_STEPS):
        d = math.exp(-decay * step / TOTAL_STEPS)

        def noise() -> torch.Tensor:
            return 1 + 0.15 * torch.randn(1, generator=generator)

        losses = {
            "loss/cls": torch.tensor(1.2 * d) * noise(),
            "loss/reg": torch.tensor(0.8 * d + 0.1) * noise(),
            "loss/heatmap": torch.tensor(0.6 * d) * noise(),
        }
        steps.append((losses, lr_at(step, base_lr)))
    return steps


class RunConfig(NamedTuple):
    """Hyperparameters and plot style for one synthetic run."""

    name: str
    base_lr: float
    decay: float
    seed: int
    color: tuple[int, int, int]


RUNS = [
    RunConfig("baseline", base_lr=1e-3, decay=3.0, seed=0, color=(31, 119, 180)),
    RunConfig("high_lr", base_lr=3e-3, decay=2.0, seed=1, color=(255, 127, 14)),
]

# %%
# Set up the dashboard
# --------------------
# We log everything to one recording and compose a single blueprint:
# :func:`~vision3d.viz.time_series_view` captures the scalars logged under each
# prefix (``train``, ``runs``, ``val``), and :func:`~vision3d.viz.lidar_view`
# captures the 3D entity tree. A row of metric plots sits above the 3D scene,
# all driven by the shared ``step`` timeline.

import rerun as rr
import rerun.blueprint as rrb

from vision3d.viz import (
    lidar_view,
    log_boxes_3d,
    log_point_cloud,
    log_scalars,
    style_series,
    time_series_view,
)

rr.init("vision3d_training", spawn=True)
rr.send_blueprint(
    rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                time_series_view(entity_prefix="train", name="loss (baseline)"),
                time_series_view(entity_prefix="runs", name="total loss (runs)"),
                time_series_view(entity_prefix="val", name="val metrics"),
            ),
            lidar_view(entity_prefix="val_sample", name="pred vs gt over training"),
            row_shares=(2, 3),
        )
    )
)
rr.log("val_sample", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

# %%
# Log per-step training metrics
# -----------------------------
# Inside the training loop, call :func:`~vision3d.viz.log_scalars` once per
# optimizer step. Names containing ``/`` (e.g. ``"loss/cls"``) nest into a
# shared entity, so the component losses group under ``train/loss``. We log the
# baseline run's full breakdown under the ``"train"`` prefix.

for step, (losses, lr) in enumerate(simulate_run(base_lr=1e-3, decay=3.0, seed=0)):
    total = sum(losses.values())
    log_scalars({"loss/total": total, **losses, "lr": lr}, step=step, prefix="train")

# %%
# Compare several runs in one plot
# --------------------------------
# Rerun overlays scalars logged to sibling entities in the same view. Routing
# each run to its own prefix (``runs/<name>``) puts their curves on one plot;
# :func:`~vision3d.viz.style_series` then gives each a stable legend name and
# color. This is the Rerun analogue of a wandb run comparison. (Rerun has no
# sweep table or parallel-coordinate view, so for experiment *management* you
# would still pair it with a tool like wandb or MLflow.)

for run in RUNS:
    style_series(
        f"runs/{run.name}/loss/total", name=run.name, color=run.color, width=2.0
    )
    steps = simulate_run(base_lr=run.base_lr, decay=run.decay, seed=run.seed)
    for step, (losses, _lr) in enumerate(steps):
        log_scalars(
            {"loss/total": sum(losses.values())},
            step=step,
            prefix=f"runs/{run.name}",
        )

# %%
# Log per-epoch validation metrics
# --------------------------------
# Validation runs less often, so log it on the ``"epoch"`` timeline (we also
# pass ``step`` so the points line up with the training curves when viewed
# against steps). In a real loop these come straight from
# :mod:`vision3d.metrics` -- e.g. ``log_scalars(metric.compute(), epoch=epoch,
# prefix="val")`` for nuScenes mAP and NDS. Here we synthesize values that
# climb as training progresses.

val_gen = torch.Generator().manual_seed(3)
for epoch in range(EPOCHS):
    progress = (epoch + 1) / EPOCHS
    jitter = 0.02 * torch.randn(1, generator=val_gen).item()
    log_scalars(
        {"mAP": 0.25 + 0.4 * progress + jitter, "NDS": 0.30 + 0.35 * progress + jitter},
        step=(epoch + 1) * STEPS_PER_EPOCH - 1,
        epoch=epoch,
        prefix="val",
    )

# %%
# Watch predictions converge on a fixed sample
# --------------------------------------------
# The most useful qualitative signal for a detector is seeing predictions snap
# onto objects as it learns. Pick a fixed validation sample; its point cloud and
# ground truth are constant, so log them once with ``static=True``. We then
# re-log the predictions on the same ``step`` timeline as the losses, so in the
# viewer the boxes converge in lockstep with the loss curves above.

from vision3d.tensors import BoundingBox3DFormat, BoundingBoxes3D

scene_gen = torch.Generator().manual_seed(7)
points = torch.empty(4000, 3)
points[:, :2] = points[:, :2].uniform_(-20, 20, generator=scene_gen)
points[:, 2] = points[:, 2].uniform_(-2, 1, generator=scene_gen)

# Three ground-truth objects (XYZLWHY: center, size, yaw).
gt = BoundingBoxes3D(
    torch.tensor(
        [
            [6.0, 2.0, 0.0, 4.5, 2.0, 1.6, 0.2],
            [-8.0, -5.0, 0.0, 4.0, 1.8, 1.5, 1.4],
            [12.0, -10.0, 0.0, 4.8, 2.1, 1.7, -0.6],
        ]
    ),
    format=BoundingBox3DFormat.XYZLWHY,
)
class_ids = [0, 0, 0]
label_to_id = {"car": 0}

log_point_cloud("val_sample/lidar", points, static=True)
log_boxes_3d(
    "val_sample/gt/boxes",
    gt,
    class_ids=class_ids,
    label_to_id=label_to_id,
    fill_mode="transparentfillmajorwireframe",
    static=True,
)

# Each object gets a fixed but badly-wrong initial guess: far-off position,
# distorted shape (0.3x-2.5x the true size), and a heading off by up to 180
# degrees. As training progresses each box converges at its *own* rate and
# wanders along a unique, decaying path -- smooth low-frequency noise rather
# than per-frame jitter -- so the predictions look like a real detector
# hunting for objects, not boxes sliding along tidy arcs. Confidence climbs
# noisily from near-zero. All of it dies down as the loss curves fall.
TWO_PI = 2 * math.pi
n = gt.shape[0]
gt_raw = gt.as_subclass(torch.Tensor)

pred_gen = torch.Generator().manual_seed(11)


def rand(*shape: int, lo: float, hi: float) -> torch.Tensor:
    """Draw a uniform tensor in ``[lo, hi)`` from the shared generator.

    Returns:
        A tensor of the requested shape.
    """
    return torch.empty(*shape).uniform_(lo, hi, generator=pred_gen)


init_offset = torch.empty(n, 3)
init_offset[:, :2] = rand(n, 2, lo=-8.0, hi=8.0)
init_offset[:, 2] = rand(n, lo=-1.5, hi=1.5)
size_scale = rand(n, 3, lo=0.3, hi=2.5)
yaw_error = rand(n, lo=-math.pi, hi=math.pi)
decay_k = rand(n, lo=2.0, hi=5.0)  # per-box convergence speed

# Smooth wander: a low-frequency sinusoid per box, per axis, with its own
# amplitude, frequency, and phase so paths curve unpredictably (no shared arc).
wander_amp = rand(n, 3, lo=0.0, hi=3.0)
wander_freq = rand(n, 3, lo=1.0, hi=3.0)
wander_phase = rand(n, 3, lo=0.0, hi=TWO_PI)
yaw_wander = rand(n, lo=0.0, hi=0.6)
yaw_freq = rand(n, lo=1.0, hi=3.0)
yaw_phase = rand(n, lo=0.0, hi=TWO_PI)

for step in range(TOTAL_STEPS):
    frac = step / TOTAL_STEPS
    alpha = 1 - torch.exp(-decay_k * frac)  # [n], per-box, 0 -> ~1
    remaining = (1 - alpha).unsqueeze(1)  # [n, 1] for broadcasting over xyz
    raw = gt_raw.clone()
    # Position: glide from the far-off guess to the GT, plus a decaying wander.
    raw[:, :3] += remaining * init_offset
    raw[:, :3] += (
        remaining * wander_amp * torch.sin(TWO_PI * wander_freq * frac + wander_phase)
    )
    # Shape: interpolate the size distortion back to the true dimensions.
    raw[:, 3:6] *= 1 + remaining * (size_scale - 1)
    # Heading: unwind the yaw error, with its own wobble.
    raw[:, 6] += (1 - alpha) * yaw_error
    raw[:, 6] += (
        (1 - alpha) * yaw_wander * torch.sin(TWO_PI * yaw_freq * frac + yaw_phase)
    )
    # Confidence climbs from near-zero as the predictions sharpen, but noisily.
    scores = (0.05 + 0.9 * alpha + 0.06 * torch.randn(n, generator=pred_gen)).clamp(
        0.02, 0.99
    )

    rr.set_time("step", sequence=step)
    log_boxes_3d(
        "val_sample/pred/boxes",
        BoundingBoxes3D(raw, format=BoundingBox3DFormat.XYZLWHY),
        class_ids=class_ids,
        label_to_id=label_to_id,
        scores=scores,
        fill_mode="majorwireframe",
        show_labels=True,
    )
