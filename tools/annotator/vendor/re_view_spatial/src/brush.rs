//! Brush selection: paint over points in the 3D view, fit a box to what was painted.
//!
//! Lives in the fork for the same reason as `box_drag`: painting needs the
//! pointer, and upstream the camera claims every mouse button with no modifier
//! escape.
//!
//! The fitted box is published for the application to consume rather than
//! written here, because naming and slot allocation are the annotator's policy,
//! not this crate's.

use ahash::HashMap;
use parking_lot::Mutex;
use re_sdk_types::archetypes::Points3D;
use re_sdk_types::components::{Position3D, Radius};
use re_viewer_context::{ViewQuery, ViewerContext};

use crate::eye::Eye;

/// Hold this and drag with the primary button to paint.
pub const BRUSH_KEY: egui::Key = egui::Key::B;
/// Brush radius in points.
const BRUSH_RADIUS: f32 = 22.0;
/// Below this many painted points a fit is meaningless.
const MIN_POINTS: usize = 8;
/// Floor on each half-extent, so a flat or single-line selection stays usable.
const MIN_HALF_EXTENT: f32 = 0.05;

/// A box fitted to a painted point set, awaiting the application.
#[derive(Clone, Copy, Debug)]
pub struct FittedBox {
    pub center: glam::Vec3,
    pub half_size: glam::Vec3,
    pub rotation: glam::Quat,
    pub points: usize,
}

static STROKE: Mutex<Vec<egui::Pos2>> = Mutex::new(Vec::new());
static PENDING: Mutex<Option<FittedBox>> = Mutex::new(None);
/// World positions of every visible point, gathered once per stroke. Re-querying
/// allocates a vector of every point in the scene, which is far too costly to
/// repeat per frame.
static CLOUD: Mutex<Vec<glam::Vec3>> = Mutex::new(Vec::new());
/// Currently covered points, as (screen, world). Feeds both the highlight and
/// the final fit, so the projection pass is done once.
static SELECTED: Mutex<Vec<(egui::Pos2, glam::Vec3)>> = Mutex::new(Vec::new());

/// Whether a stroke is in progress, in which case the camera must ignore the
/// pointer.
pub fn is_active() -> bool {
    !STROKE.lock().is_empty()
}

/// Take the most recent fit, if a stroke finished since the last call.
pub fn take_pending() -> Option<FittedBox> {
    PENDING.lock().take()
}

/// Drive painting for this frame.
pub fn handle(
    ctx: &ViewerContext<'_>,
    ui: &egui::Ui,
    response: &egui::Response,
    eye: &Eye,
    rect: egui::Rect,
    query: &ViewQuery<'_>,
) {
    let (key_down, primary_down) =
        ui.input(|i| (i.key_down(BRUSH_KEY), i.pointer.primary_down()));
    let pointer = response.interact_pointer_pos().or_else(|| response.hover_pos());

    // Sweeps are stored in map coordinates while the view renders in ego, so the
    // projection has to fold in ego_from_map. The fitted box then needs no
    // conversion, being map-framed already.
    let ego_from_map = crate::frames::map_from_ego(ctx).inverse();

    if !key_down {
        // Releasing the key finishes any stroke in flight.
        finish();
        return;
    }

    let painter = ui.painter_at(rect);

    // Dim the whole view while the brush is armed, so the highlighted points
    // read as "selected" against everything else.
    painter.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(140));

    if primary_down {
        let mut stroke = STROKE.lock();
        let grew = if let Some(p) = pointer {
            // Sample sparsely: closer samples cost work without adding coverage.
            if stroke
                .last()
                .is_none_or(|last| last.distance(p) > BRUSH_RADIUS * 0.4)
            {
                stroke.push(p);
                true
            } else {
                false
            }
        } else {
            false
        };

        if grew {
            if CLOUD.lock().is_empty() {
                *CLOUD.lock() = gather_visible_points(ctx, query);
            }
            let cloud = CLOUD.lock();
            *SELECTED.lock() = select_points(&cloud, eye, rect, ego_from_map, &stroke);
        }
        drop(stroke);

        // The covered points, drawn over the veil.
        let selected = SELECTED.lock();
        painter.extend(selected.iter().map(|(pos, _)| {
            egui::Shape::rect_filled(
                egui::Rect::from_center_size(*pos, egui::Vec2::splat(3.0)),
                0.0,
                egui::Color32::from_rgb(255, 214, 92),
            )
        }));
    }

    if let Some(p) = pointer {
        painter.circle_stroke(
            p,
            BRUSH_RADIUS,
            egui::Stroke::new(1.5, egui::Color32::from_white_alpha(200)),
        );
    }

    if !primary_down {
        // Armed but not pressed: the previous stroke, if any, is done.
        finish();
    }
}

/// Fit whatever was covered, and drop the per-stroke caches.
fn finish() {
    let had_stroke = !STROKE.lock().is_empty();
    if !had_stroke {
        CLOUD.lock().clear();
        SELECTED.lock().clear();
        return;
    }

    let picked: Vec<glam::Vec3> = SELECTED.lock().iter().map(|(_, world)| *world).collect();
    STROKE.lock().clear();
    CLOUD.lock().clear();
    SELECTED.lock().clear();

    match fit_yaw_aligned(&picked) {
        Some(fit) => {
            re_log::info!(
                "brush fitted a box to {} points: size {:?}",
                fit.points,
                fit.half_size * 2.0
            );
            *PENDING.lock() = Some(fit);
        }
        None => re_log::info!(
            "brush covered {} points, too few to fit a box",
            picked.len()
        ),
    }
}

#[expect(dead_code)]
fn paint_preview(ui: &egui::Ui, rect: egui::Rect, stroke: &[egui::Pos2], pointer: Option<egui::Pos2>) {
    let painter = ui.painter_at(rect);
    for p in stroke {
        painter.circle_filled(*p, BRUSH_RADIUS, egui::Color32::from_white_alpha(16));
    }
    if let Some(p) = pointer {
        painter.circle_stroke(
            p,
            BRUSH_RADIUS,
            egui::Stroke::new(1.5, egui::Color32::from_white_alpha(160)),
        );
    }
}

/// Every point currently drawn, in world space.
///
/// Clouds hidden by the annotator's sweeps slider carry a zero radius; skipping
/// them keeps the brush to what is actually on screen.
fn gather_visible_points(ctx: &ViewerContext<'_>, query: &ViewQuery<'_>) -> Vec<glam::Vec3> {
    re_tracing::profile_function!();

    let positions_id = Points3D::descriptor_positions().component;
    let radii_id = Points3D::descriptor_radii().component;
    let at = ctx.current_query();
    let recording = ctx.recording();

    // Not restricted to the view origin: sweeps live in the map frame, outside
    // the ego subtree this view is rooted at, and filtering by origin left the
    // brush with nothing to select.
    let _ = query;
    let mut out = Vec::new();
    for path in recording.sorted_entity_paths().cloned().collect::<Vec<_>>() {
        let results = recording.latest_at(&at, &path, [positions_id, radii_id]);

        if results
            .component_batch::<Radius>(radii_id)
            .and_then(|radii| radii.first().copied())
            .is_some_and(|r| r.0.0 == 0.0)
        {
            continue;
        }

        if let Some(positions) = results.component_batch::<Position3D>(positions_id) {
            out.extend(positions.into_iter().map(|p| glam::Vec3::from_array(p.0.0)));
        }
    }
    out
}

/// Project the cloud and keep the points the stroke covers.
fn select_points(
    cloud: &[glam::Vec3],
    eye: &Eye,
    rect: egui::Rect,
    ego_from_map: glam::Affine3A,
    samples: &[egui::Pos2],
) -> Vec<(egui::Pos2, glam::Vec3)> {
    re_tracing::profile_function!();

    // Bucket stroke samples so each point only tests nearby ones.
    let cell = BRUSH_RADIUS;
    let mut buckets: HashMap<(i32, i32), Vec<egui::Pos2>> = HashMap::default();
    for s in samples {
        let key = ((s.x / cell).floor() as i32, (s.y / cell).floor() as i32);
        buckets.entry(key).or_default().push(*s);
    }

    // Fold the frame change into the projection rather than transforming every
    // point separately.
    let ui_from_map = eye.ui_from_world(rect) * glam::Mat4::from(ego_from_map);
    let mut picked = Vec::new();

    for world in cloud {
        let clip = ui_from_map * world.extend(1.0);
        if clip.w <= 0.0 {
            continue; // behind the eye
        }
        let screen = egui::pos2(clip.x / clip.w, clip.y / clip.w);
        if !rect.contains(screen) {
            continue;
        }

        let key = ((screen.x / cell).floor() as i32, (screen.y / cell).floor() as i32);
        let hit = (-1..=1).any(|dx| {
            (-1..=1).any(|dy| {
                buckets
                    .get(&(key.0 + dx, key.1 + dy))
                    .is_some_and(|near| near.iter().any(|s| s.distance(screen) <= BRUSH_RADIUS))
            })
        });
        if hit {
            picked.push((screen, *world));
        }
    }
    picked
}

/// Fit a yaw-only oriented box to a point set.
///
/// Orientation comes from the principal axis of the points' XY covariance, which
/// for ground objects is what you want: a full 3D PCA happily returns a box
/// tilted by whatever noise dominates the vertical spread. Extents are then the
/// bounds in that rotated frame.
fn fit_yaw_aligned(points: &[glam::Vec3]) -> Option<FittedBox> {
    if points.len() < MIN_POINTS {
        return None;
    }

    let n = points.len() as f32;
    let centroid = points.iter().copied().fold(glam::Vec3::ZERO, |a, b| a + b) / n;

    let (mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32);
    for p in points {
        let d = *p - centroid;
        sxx += d.x * d.x;
        sxy += d.x * d.y;
        syy += d.y * d.y;
    }

    let yaw = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let rotation = glam::Quat::from_rotation_z(yaw);
    let to_local = rotation.inverse();

    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for p in points {
        let local = to_local * (*p - centroid);
        min = min.min(local);
        max = max.max(local);
    }

    let half_size = ((max - min) * 0.5).max(glam::Vec3::splat(MIN_HALF_EXTENT));
    let center = centroid + rotation * ((min + max) * 0.5);

    Some(FittedBox {
        center,
        half_size,
        rotation,
        points: points.len(),
    })
}
