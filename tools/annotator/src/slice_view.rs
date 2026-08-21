//! The box-frame slice views: three orthographic 2D views locked to the
//! selected box's own coordinate frame.
//!
//! Because the view is expressed in the box's frame, the box is an
//! axis-aligned rectangle, so every manipulation is a plain 2D mouse gesture:
//! drag the body to translate, drag an edge to resize, drag the handle to
//! rotate. Three views cover all 9 degrees of freedom.
//!
//! While a drag is in flight the view frame is *frozen* at the pose the box had
//! when the drag started. Without that the box would stay pinned to the centre
//! of its own view and dragging it would look like nothing was happening.

use std::marker::PhantomData;

use glam::Vec3;
use rerun::external::egui;
use rerun::external::re_chunk::{Chunk, TimePoint};
use rerun::external::re_log;
use rerun::external::re_log_types::EntityPath;
use rerun::external::re_sdk_types::ViewClassIdentifier;
use rerun::external::re_ui::{self, Help};
use rerun::external::re_viewer_context::{
    IdentifiedViewSystem as _, IndicatedEntities, MissingChunkReporter, PerVisualizerType,
    RecommendedVisualizers, SystemCommand, SystemCommandSender as _, SystemExecutionOutput,
    ViewClass, ViewClassLayoutPriority, ViewClassRegistryError, ViewClassUiOutput, ViewQuery,
    ViewSpawnHeuristics, ViewState, ViewSystemExecutionError, ViewSystemIdentifier,
    ViewSystemRegistrator, ViewerContext, VisualizableReason,
};

use crate::box_edit::{
    Box9Dof, DragKind, Edge, Frame, ROTATE_HANDLE_OFFSET, SliceAxis, apply_drag, hit_test,
};
use crate::slice_visualizer::{
    SliceBox, SliceBoxesOutput, SliceBoxesVisualizer, SlicePoints, SlicePointsOutput,
    SlicePointsVisualizer,
};

/// How much of the box's own extent to show around it, as a multiplier.
const DEFAULT_ZOOM: f32 = 2.5;
/// How far outside the box, along the view axis, points are still drawn.
const SLAB_MARGIN: f32 = 0.35;
/// Pointer grab radius for edges and the rotate handle, in points.
const GRAB_RADIUS_PX: f32 = 7.0;
/// Upper bound on points painted per view, to keep the UI responsive on dense
/// clouds. Points are dropped by stride, so coverage stays uniform.
const MAX_POINTS_DRAWN: usize = 20_000;

/// Compile-time marker so each slice axis gets its own `ViewClass` type, which
/// is what `App::add_view_class::<T>()` requires.
pub trait AxisMarker: 'static + Send + Sync + Default {
    const AXIS: SliceAxis;
}

#[derive(Default)]
pub struct AxisZ;
#[derive(Default)]
pub struct AxisY;
#[derive(Default)]
pub struct AxisX;

impl AxisMarker for AxisZ {
    const AXIS: SliceAxis = SliceAxis::Z;
}
impl AxisMarker for AxisY {
    const AXIS: SliceAxis = SliceAxis::Y;
}
impl AxisMarker for AxisX {
    const AXIS: SliceAxis = SliceAxis::X;
}

/// A drag in progress. Everything is captured at drag start so the gesture is
/// applied absolutely rather than accumulated frame by frame.
struct ActiveDrag {
    entity: EntityPath,
    start_box: Box9Dof,
    anchor: Frame,
    kind: DragKind,
    start_u: f32,
    start_v: f32,
    /// Pixels per scene unit, frozen for the duration of the drag.
    ///
    /// The live scale is derived from the box's own extent so the box fills the
    /// view. Recomputing it mid-drag makes resizing self-defeating: the box
    /// grows, the view zooms out to fit, the pointer's plane coordinate shrinks,
    /// and the gesture settles at a fixed point where nothing moves.
    scale: f32,
}

#[derive(Default)]
pub struct SliceViewState {
    drag: Option<ActiveDrag>,
    zoom: Option<f32>,
}

impl ViewState for SliceViewState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn heap_size_bytes(&self) -> u64 {
        0
    }
}

/// Maps between view-plane scene units and screen space.
struct Projection {
    center: egui::Pos2,
    scale: f32,
}

impl Projection {
    fn to_screen(&self, u: f32, v: f32) -> egui::Pos2 {
        // Screen y grows downwards; the view plane's v grows upwards.
        egui::pos2(self.center.x + u * self.scale, self.center.y - v * self.scale)
    }

    fn to_plane(&self, p: egui::Pos2) -> (f32, f32) {
        (
            (p.x - self.center.x) / self.scale,
            (self.center.y - p.y) / self.scale,
        )
    }
}

#[derive(Default)]
pub struct BoxSliceView<A: AxisMarker>(PhantomData<A>);

impl<A: AxisMarker> ViewClass for BoxSliceView<A> {
    fn identifier() -> ViewClassIdentifier {
        A::AXIS.identifier().into()
    }

    fn display_name(&self) -> &'static str {
        A::AXIS.display_name()
    }

    fn icon(&self) -> &'static re_ui::Icon {
        &re_ui::icons::VIEW_GENERIC
    }

    fn help(&self, _os: egui::os::OperatingSystem) -> Help {
        Help::new("Box slice").markdown(
            "Edits the selected 3D box in its own coordinate frame.\n\n\
             - drag inside the rectangle to move it\n\
             - drag an edge to resize that face\n\
             - drag the handle to rotate about this view's axis\n\
             - scroll to zoom",
        )
    }

    fn on_register(
        &self,
        system_registry: &mut ViewSystemRegistrator<'_>,
    ) -> Result<(), ViewClassRegistryError> {
        system_registry.register_visualizer::<SlicePointsVisualizer>()?;
        system_registry.register_visualizer::<SliceBoxesVisualizer>()
    }

    fn new_state(&self) -> Box<dyn ViewState> {
        Box::<SliceViewState>::default()
    }

    fn layout_priority(&self) -> ViewClassLayoutPriority {
        Default::default()
    }

    fn spawn_heuristics(
        &self,
        ctx: &ViewerContext<'_>,
        include_entity: &dyn Fn(&EntityPath) -> bool,
    ) -> ViewSpawnHeuristics {
        // Only worth spawning if there is actually something to edit.
        if ctx
            .visualizable_entities_per_visualizer
            .get(&SliceBoxesVisualizer::identifier())
            .is_some_and(|entities| entities.keys().any(include_entity))
        {
            ViewSpawnHeuristics::root()
        } else {
            ViewSpawnHeuristics::empty()
        }
    }

    fn recommended_visualizers_for_entity(
        &self,
        _entity_path: &EntityPath,
        visualizers: &[(ViewSystemIdentifier, &VisualizableReason)],
        _indicated: &PerVisualizerType<&IndicatedEntities>,
    ) -> RecommendedVisualizers {
        // Take everything we can render; the view needs points and boxes
        // together and neither logs an indicator we could key off.
        RecommendedVisualizers::default_many(visualizers.iter().map(|(viz, _)| *viz))
    }

    fn ui(
        &self,
        ctx: &ViewerContext<'_>,
        _missing_chunk_reporter: &MissingChunkReporter,
        ui: &mut egui::Ui,
        state: &mut dyn ViewState,
        query: &ViewQuery<'_>,
        system_output: SystemExecutionOutput,
    ) -> Result<ViewClassUiOutput, ViewSystemExecutionError> {
        let clouds = system_output
            .visualizer_data_or_default::<SlicePointsOutput>(SlicePointsVisualizer::identifier())?;
        let boxes = system_output
            .visualizer_data_or_default::<SliceBoxesOutput>(SliceBoxesVisualizer::identifier())?;
        let clouds: &[SlicePoints] = clouds.as_ref();
        let boxes: &[SliceBox] = boxes.as_ref();

        let state = state
            .as_any_mut()
            .downcast_mut::<SliceViewState>()
            .ok_or(ViewSystemExecutionError::StateCastError("SliceViewState"))?;

        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

        // Pick the box to edit: whichever is selected, else the only one.
        let selected = ctx
            .selection()
            .iter_items()
            .filter_map(|item| item.entity_path())
            .find(|p| boxes.iter().any(|b| &b.entity == *p));

        let Some(active) = selected
            .and_then(|p| boxes.iter().find(|b| &b.entity == p))
            .or_else(|| (boxes.len() == 1).then(|| boxes.first()).flatten())
        else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Select a box to edit",
                egui::FontId::proportional(13.0),
                ui.visuals().weak_text_color(),
            );
            return Ok(Default::default());
        };

        let axis = A::AXIS;
        let (iu, iv, in_) = axis.axes();

        // Sweeps and boxes are both stored in the map frame, so these views need
        // no conversion: the anchor and the points already share a frame.
        let active_bbox = active.bbox;

        // Freeze the view frame while dragging so the box visibly moves.
        let anchor = match &state.drag {
            Some(drag) if drag.entity == active.entity => drag.anchor,
            _ => active_bbox.frame(),
        };

        let zoom = state.zoom.get_or_insert(DEFAULT_ZOOM);
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                *zoom = (*zoom * (1.0 - scroll * 0.002)).clamp(1.05, 40.0);
            }
        }

        let hu = active_bbox.half_size[iu];
        let hv = active_bbox.half_size[iv];
        let scale = match &state.drag {
            Some(drag) if drag.entity == active.entity => drag.scale,
            _ => (rect.width() / (2.0 * hu * *zoom)).min(rect.height() / (2.0 * hv * *zoom)),
        };
        let proj = Projection {
            center: rect.center(),
            scale,
        };

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);

        // --- points -------------------------------------------------------
        // Honour the sweeps slider: these views paint their own points, so the
        // zero-radius trick the 3D view relies on has no effect here.
        let visible: Vec<&SlicePoints> = clouds
            .iter()
            .filter(|cloud| crate::settings::cloud_visible(&cloud.entity))
            .collect();

        // Decimate on what actually lands in the slab, not on the whole cloud.
        // Striding by total point count would cancel the sweeps slider out: more
        // sweeps means a bigger total, hence a coarser stride, hence the same
        // number of points drawn.
        let half_n = active_bbox.half_size[in_] + SLAB_MARGIN;
        let point_px = (crate::settings::point_radius() * 2.0 * scale).clamp(1.0, 14.0);

        let mut in_slab: Vec<(egui::Pos2, egui::Color32)> = Vec::new();
        let default_color = ui.visuals().text_color().gamma_multiply(0.55);
        for cloud in visible {
            for (i, p) in cloud.positions.iter().enumerate() {
                let local = anchor.to_local(*p);
                if local[in_].abs() > half_n {
                    continue;
                }
                let pos = proj.to_screen(local[iu], local[iv]);
                if !rect.contains(pos) {
                    continue;
                }
                in_slab.push((pos, cloud.colors.get(i).copied().unwrap_or(default_color)));
            }
        }

        let stride = (in_slab.len() / MAX_POINTS_DRAWN).max(1);
        let shapes: Vec<egui::Shape> = in_slab
            .iter()
            .step_by(stride)
            .map(|(pos, color)| {
                egui::Shape::rect_filled(
                    egui::Rect::from_center_size(*pos, egui::Vec2::splat(point_px)),
                    0.0,
                    *color,
                )
            })
            .collect();
        painter.extend(shapes);

        // --- the box ------------------------------------------------------
        // Drawn from the box's own frame, so a rotation drag shows up as a
        // rotated rectangle against the frozen anchor.
        let box_frame = active_bbox.frame();
        let corner = |su: f32, sv: f32| {
            let mut local = Vec3::ZERO;
            local[iu] = su * active_bbox.half_size[iu];
            local[iv] = sv * active_bbox.half_size[iv];
            let world = box_frame.to_world(local);
            let in_anchor = anchor.to_local(world);
            proj.to_screen(in_anchor[iu], in_anchor[iv])
        };

        let corners = [
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, 1.0),
        ];
        // Match the 3D view: the class colour, falling back to the UI accent for
        // boxes with no class yet.
        let accent = crate::ontology::class_color(ctx, active.class_id)
            .unwrap_or_else(|| ui.visuals().selection.bg_fill);
        painter.add(egui::Shape::closed_line(
            corners.to_vec(),
            egui::Stroke::new(1.5, accent),
        ));

        // Rotation handle, on the +v side.
        let handle = {
            let mut local = Vec3::ZERO;
            local[iv] = active_bbox.half_size[iv] * ROTATE_HANDLE_OFFSET;
            let in_anchor = anchor.to_local(box_frame.to_world(local));
            proj.to_screen(in_anchor[iu], in_anchor[iv])
        };
        painter.line_segment(
            [(corners[2] + corners[3].to_vec2()) / 2.0, handle],
            egui::Stroke::new(1.0, accent.gamma_multiply(0.6)),
        );
        painter.circle_filled(handle, 4.0, accent);

        // --- interaction --------------------------------------------------
        let tol = GRAB_RADIUS_PX / scale;

        // Show which manipulation the pointer is over, so the mode is legible
        // before committing to a drag.
        let hover_kind = state.drag.as_ref().map(|d| d.kind).or_else(|| {
            response
                .hover_pos()
                .map(|p| proj.to_plane(p))
                .and_then(|(u, v)| hit_test(u, v, hu, hv, tol))
        });

        if let Some(kind) = hover_kind {
            ui.ctx().set_cursor_icon(cursor_for(kind, axis));
        }


        // egui has no rotate cursor, so draw the affordance instead.
        if hover_kind == Some(DragKind::Rotate) {
            paint_rotation_glyph(&painter, handle, 11.0, accent);
        }

        if response.drag_started()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let (u, v) = proj.to_plane(pointer);
            if let Some(kind) = hit_test(u, v, hu, hv, tol) {
                state.drag = Some(ActiveDrag {
                    entity: active.entity.clone(),
                    start_box: active_bbox,
                    anchor,
                    kind,
                    start_u: u,
                    start_v: v,
                    scale,
                });
            }
        }

        if response.drag_stopped() {
            state.drag = None;
        }

        if let Some(drag) = &state.drag
            && response.dragged()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let (u, v) = proj.to_plane(pointer);
            let angle = if drag.kind == DragKind::Rotate {
                v.atan2(u) - drag.start_v.atan2(drag.start_u)
            } else {
                0.0
            };

            let edited = apply_drag(
                &drag.start_box,
                axis,
                drag.kind,
                u - drag.start_u,
                v - drag.start_v,
                angle,
            );

            if edited != active_bbox {
                write_box(ctx, query, &drag.entity, &edited);
            }
        }

        Ok(Default::default())
    }
}

/// Which cursor communicates a given manipulation.
///
/// The in-plane axes map to screen axes directly (`u` horizontal, `v` vertical),
/// so a `u` face resizes horizontally and a `v` face vertically regardless of
/// which slice axis this view is showing.
fn cursor_for(kind: DragKind, _axis: SliceAxis) -> egui::CursorIcon {
    match kind {
        DragKind::Body => egui::CursorIcon::Move,
        DragKind::Edge(Edge::MinU | Edge::MaxU) => egui::CursorIcon::ResizeHorizontal,
        DragKind::Edge(Edge::MinV | Edge::MaxV) => egui::CursorIcon::ResizeVertical,
        // No circular-arrow cursor exists in egui; `paint_rotation_glyph` draws
        // the affordance and this is the closest "free movement" cursor.
        DragKind::Rotate => egui::CursorIcon::AllScroll,
    }
}

/// Draw a circular arrow: an arc with an arrowhead, marking a rotation handle.
fn paint_rotation_glyph(
    painter: &egui::Painter,
    center: egui::Pos2,
    radius: f32,
    color: egui::Color32,
) {
    use std::f32::consts::TAU;

    // Three-quarter arc, leaving a gap so it reads as an arrow rather than a ring.
    const START: f32 = 0.15 * TAU;
    const SWEEP: f32 = 0.72 * TAU;
    const SEGMENTS: usize = 24;

    let points: Vec<egui::Pos2> = (0..=SEGMENTS)
        .map(|i| {
            let a = START + SWEEP * (i as f32 / SEGMENTS as f32);
            egui::pos2(center.x + radius * a.cos(), center.y + radius * a.sin())
        })
        .collect();

    let tip = *points.last().expect("arc always has points");
    painter.add(egui::Shape::line(points, egui::Stroke::new(1.6, color)));

    // Arrowhead tangent to the arc at its end.
    let end_angle = START + SWEEP;
    let tangent = egui::vec2(-end_angle.sin(), end_angle.cos());
    let normal = egui::vec2(-tangent.y, tangent.x);
    let head = 4.0;
    painter.add(egui::Shape::convex_polygon(
        vec![
            tip + tangent * head,
            tip - tangent * head * 0.3 + normal * head * 0.7,
            tip - tangent * head * 0.3 - normal * head * 0.7,
        ],
        color,
        egui::Stroke::NONE,
    ));
}

/// Write an edited box back to the recording at the current time.
///
/// Rerun treats recordings as append-only, so this adds a new row at the
/// current time rather than mutating the original. Latest-at semantics mean
/// the edit is what every view then reads.
fn write_box(ctx: &ViewerContext<'_>, query: &ViewQuery<'_>, entity: &EntityPath, b: &Box9Dof) {
    let Some(timeline) = ctx.recording().timelines().get(&query.timeline).copied() else {
        return;
    };

    let archetype = rerun::Boxes3D::from_centers_and_half_sizes(
        [(b.center.x, b.center.y, b.center.z)],
        [(b.half_size.x, b.half_size.y, b.half_size.z)],
    )
    .with_quaternions([rerun::Quaternion::from_xyzw(b.rotation.to_array())]);

    // A static box must keep taking static writes, or the edit would be shadowed
    // by the static row it already has.
    let timepoint = re_view_spatial_fork::static_boxes::timepoint_for(
        entity,
        TimePoint::from([(timeline, query.latest_at)]),
    );

    let chunk = Chunk::builder(entity.clone())
        .with_archetype_auto_row(timepoint, &archetype)
        .build();

    match chunk {
        Ok(chunk) => ctx
            .command_sender()
            .send_system(SystemCommand::AppendToStore(
                ctx.store_id().clone(),
                vec![chunk],
            )),
        Err(err) => re_log::error_once!("failed to build box edit chunk: {err}"),
    }
}
