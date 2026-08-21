//! Click-and-drag manipulation of 3D boxes: move a face, resize an edge.
//!
//! This is the reason this crate is vendored. Upstream, every mouse button in
//! the 3D view belongs to the camera (`ROTATE3D_BUTTON` is the primary button,
//! with no modifier escape) and a view class cannot intercept another view's
//! input, so this cannot be implemented from outside the crate.
//!
//! Drag state lives in a process-global rather than in `SpatialViewState`,
//! purely to keep the diff against upstream small: there is one annotation 3D
//! view in practice, and threading a new field through the state struct would
//! touch its `SizeBytes` derivation and every construction site.

use parking_lot::Mutex;
use re_log_types::EntityPath;
use re_sdk_types::archetypes::Boxes3D;
use re_sdk_types::components::{HalfSize3D, Position3D, RotationQuat};
use re_viewer_context::{SystemCommand, SystemCommandSender as _, ViewerContext};

use crate::eye::Eye;
use crate::ui::SpatialViewState;

/// How close to a box's limit along an axis counts as "on that face", as a
/// fraction of the half-extent. A second axis also within this band means the
/// pointer is near an edge rather than in the middle of a face.
const FACE_BAND: f32 = 0.75;
/// Smallest half-extent a resize may produce, in scene units.
const MIN_HALF_SIZE: f32 = 0.025;

/// What a drag does, decided once at press time.
#[derive(Clone, Copy)]
enum Mode {
    /// Grabbed the middle of a face: translate the whole box.
    Move,
    /// Grabbed near an edge: move the adjacent face along its own local axis,
    /// which changes size and re-centres so the opposite face stays put.
    Resize { axis: usize, sign: f32 },
}

#[derive(Clone)]
struct BoxDrag {
    entity: EntityPath,
    mode: Mode,
    /// Where the gesture started, in world space.
    grab_point: glam::Vec3,
    /// Pose at press time. Drags are applied as an absolute offset from this,
    /// so they cannot accumulate drift.
    start_center: glam::Vec3,
    start_half: glam::Vec3,
    rotation: glam::Quat,
}

static ACTIVE: Mutex<Option<BoxDrag>> = Mutex::new(None);

/// Whether a box drag is in flight, in which case the camera must ignore the
/// pointer so the box does not move and the view spin at the same time.
pub fn is_active() -> bool {
    ACTIVE.lock().is_some()
}

/// Drive box dragging for this frame. Call after picking has run, so that
/// `state.previous_picking_result` refers to the current pointer position.
pub fn handle(
    ctx: &ViewerContext<'_>,
    ui: &egui::Ui,
    response: &egui::Response,
    eye: &Eye,
    rect: egui::Rect,
    state: &SpatialViewState,
) {
    let (primary_down, primary_pressed) =
        ui.input(|i| (i.pointer.primary_down(), i.pointer.primary_pressed()));

    if !primary_down {
        *ACTIVE.lock() = None;
        // Preview what a press would do, so the two modes are discoverable.
        if let Some(pose) = hovered_box(ctx, state) {
            let mode = classify(&pose, hovered_point(state), eye);
            ui.ctx().set_cursor_icon(match mode {
                Some(Mode::Move) => egui::CursorIcon::Move,
                Some(Mode::Resize { .. }) => egui::CursorIcon::ResizeHorizontal,
                None => egui::CursorIcon::Default,
            });
        }
        return;
    }

    let Some(pointer) = response.interact_pointer_pos().or_else(|| response.hover_pos()) else {
        return;
    };

    // Picking and rays come out of the view in ego coordinates; box poses come
    // out of the store in map coordinates. Lift the ego side into map and do all
    // the arithmetic there, which is also the frame the edit is written in.
    let map_from_ego = crate::frames::map_from_ego(ctx);

    let mut active = ACTIVE.lock();

    if active.is_none() {
        if !primary_pressed {
            // Mid-drag of something that wasn't a box (usually the camera).
            return;
        }
        let (Some(pose), Some(grab_point)) = (
            hovered_box(ctx, state),
            hovered_point(state).map(|p| map_from_ego.transform_point3(p)),
        ) else {
            return;
        };
        let Some(mode) = classify(&pose, Some(grab_point), eye) else {
            return;
        };

        *active = Some(BoxDrag {
            entity: pose.entity,
            mode,
            grab_point,
            start_center: pose.center,
            start_half: pose.half,
            rotation: pose.rotation,
        });
        return;
    }

    let Some(drag) = active.clone() else {
        return;
    };

    let ego_ray = eye.picking_ray(rect, glam::vec2(pointer.x, pointer.y));
    let ray = macaw::Ray3::from_origin_dir(
        map_from_ego.transform_point3(ego_ray.origin),
        map_from_ego.transform_vector3(ego_ray.dir).normalize(),
    );

    match drag.mode {
        Mode::Move => {
            // Slide along the horizontal plane through the grab point, so a box
            // stays on the surface it was picked from rather than drifting
            // toward or away from the camera.
            if ray.dir.z.abs() < 1e-6 {
                return;
            }
            let t = (drag.grab_point.z - ray.origin.z) / ray.dir.z;
            if t <= 0.0 {
                return;
            }
            let hit = ray.origin + ray.dir * t;
            let center = drag.start_center + (hit - drag.grab_point);
            write(ctx, &drag.entity, Some(center), None);
        }

        Mode::Resize { axis, sign } => {
            // Resizing has to work for the top and bottom faces too, where a
            // horizontal plane is nearly parallel to the ray. A plane facing the
            // camera always has a well-conditioned intersection.
            let normal = map_from_ego
                .transform_vector3(eye.forward_in_world())
                .normalize();
            let denom = ray.dir.dot(normal);
            if denom.abs() < 1e-6 {
                return;
            }
            let t = (drag.grab_point - ray.origin).dot(normal) / denom;
            if t <= 0.0 {
                return;
            }
            let hit = ray.origin + ray.dir * t;

            // Only motion along the grabbed face's own outward normal counts.
            let outward = sign * (drag.rotation * unit(axis));
            let along = (hit - drag.grab_point).dot(outward);

            // Moving a face by `along` changes the half-extent by half that and
            // shifts the centre by the same amount, leaving the opposite face
            // where it was. Deriving the shift from the clamped half-extent keeps
            // that true when the box bottoms out.
            let old_half = drag.start_half[axis];
            let new_half = (old_half + along * 0.5).max(MIN_HALF_SIZE);

            let mut half = drag.start_half;
            half[axis] = new_half;
            let center = drag.start_center + outward * (new_half - old_half);

            write(ctx, &drag.entity, Some(center), Some(half));
        }
    }
}

fn unit(axis: usize) -> glam::Vec3 {
    let mut v = glam::Vec3::ZERO;
    v[axis] = 1.0;
    v
}

struct BoxPose {
    entity: EntityPath,
    center: glam::Vec3,
    half: glam::Vec3,
    rotation: glam::Quat,
}

/// The world position under the pointer, as computed by this view's picking.
fn hovered_point(state: &SpatialViewState) -> Option<glam::Vec3> {
    state
        .previous_picking_result
        .as_ref()
        .and_then(|result| result.hits.first())
        .map(|hit| hit.space_position)
}

/// The box under the pointer, if the picked entity is one.
fn hovered_box(ctx: &ViewerContext<'_>, state: &SpatialViewState) -> Option<BoxPose> {
    let hit = state
        .previous_picking_result
        .as_ref()
        .and_then(|result| result.hits.first())?;
    let entity = ctx
        .recording()
        .entity_path_from_hash(&hit.instance_path_hash.entity_path_hash)?
        .clone();

    let query = ctx.current_query();
    let db = ctx.recording();

    let ((half_time, _), half) = db.latest_at_component::<HalfSize3D>(
        &entity,
        &query,
        Boxes3D::descriptor_half_sizes().component,
    )?;
    if half_time == re_log_types::TimeInt::STATIC {
        crate::static_boxes::set_static(&entity, true);
    }
    let center = db
        .latest_at_component::<Position3D>(&entity, &query, Boxes3D::descriptor_centers().component)
        .map_or(glam::Vec3::ZERO, |(_, c)| c.0.into());
    let rotation = db
        .latest_at_component::<RotationQuat>(
            &entity,
            &query,
            Boxes3D::descriptor_quaternions().component,
        )
        .map_or(glam::Quat::IDENTITY, |(_, q)| glam::Quat::from_array(q.0.0));

    Some(BoxPose {
        entity,
        center,
        half: half.0.into(),
        rotation,
    })
}

/// Decide whether a press moves the box or resizes one of its faces.
///
/// The grab point is expressed in the box's own frame and compared against its
/// half-extents. The axis nearest its limit is the face that was clicked; if a
/// *second* axis is also near its limit the pointer is by an edge, and that
/// second face is the one the drag resizes. Grabbing the middle of a face moves
/// the whole box.
fn classify(pose: &BoxPose, grab: Option<glam::Vec3>, _eye: &Eye) -> Option<Mode> {
    // `grab` must already be in the same frame as `pose` (map).
    let grab = grab?;
    let local = pose.rotation.inverse() * (grab - pose.center);

    let mut ratios: Vec<(usize, f32)> = (0..3)
        .map(|i| (i, local[i].abs() / pose.half[i].max(1e-6)))
        .collect();
    ratios.sort_by(|a, b| b.1.total_cmp(&a.1));

    let (_, primary_ratio) = ratios[0];
    let (second_axis, second_ratio) = ratios[1];

    if primary_ratio < FACE_BAND {
        // Not near the surface at all; treat as a body grab.
        return Some(Mode::Move);
    }

    if second_ratio >= FACE_BAND {
        Some(Mode::Resize {
            axis: second_axis,
            sign: local[second_axis].signum(),
        })
    } else {
        Some(Mode::Move)
    }
}

/// Write back only the fields that changed, leaving orientation and class alone.
fn write(
    ctx: &ViewerContext<'_>,
    entity: &EntityPath,
    center: Option<glam::Vec3>,
    half: Option<glam::Vec3>,
) {
    let query = ctx.current_query();
    // A static query has no timeline to write back onto.
    let Some(timeline_name) = query.timeline() else {
        return;
    };
    let Some(timeline) = ctx.recording().timelines().get(&timeline_name).copied() else {
        return;
    };

    let mut archetype = Boxes3D::update_fields();
    if let Some(c) = center {
        archetype = archetype.with_centers([(c.x, c.y, c.z)]);
    }
    if let Some(h) = half {
        archetype = archetype.with_half_sizes([(h.x, h.y, h.z)]);
    }

    let timepoint = crate::static_boxes::timepoint_for(
        entity,
        re_chunk::TimePoint::from([(timeline, query.at())]),
    );

    match re_chunk::Chunk::builder(entity.clone())
        .with_archetype_auto_row(timepoint, &archetype)
        .build()
    {
        Ok(chunk) => ctx
            .command_sender()
            .send_system(SystemCommand::AppendToStore(
                ctx.store_id().clone(),
                vec![chunk],
            )),
        Err(err) => re_log::error_once!("failed to build box drag chunk: {err}"),
    }
}
