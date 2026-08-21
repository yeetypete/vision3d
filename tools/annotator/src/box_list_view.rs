//! Overview of every box in the frame, with label editing and box creation.
//!
//! This is the view that answers "what is in this frame": a flat list of all
//! boxes, their class, and their distance from the sensor origin. Clicking a row
//! selects that box, which is what the slice views lock onto, so this doubles as
//! the navigation surface for annotating a frame box by box.
//!
//! The label dropdown is populated from the `AnnotationContext` in the
//! recording, so the available classes are exactly the dataset's ontology rather
//! than anything hard-coded here.
//!
//! New boxes are placed by pointing at the target in the built-in 3D view and
//! pressing `N` (or clicking "New box", which uses the last hovered point). The
//! position comes from `ItemContext::ThreeD`, which the 3D view fills in with
//! the picked world position each frame the pointer is over it. That means a new
//! box lands on actual geometry rather than floating in space, which is what you
//! want when annotating a lidar cloud.

use std::sync::atomic::{AtomicU64, Ordering};

use glam::Vec3;
use rerun::external::egui;
use rerun::external::re_chunk::{Chunk, RowId, TimePoint};
use rerun::external::re_entity_db::InstancePath;
use rerun::external::re_log;
use rerun::external::re_log_types::EntityPath;
use rerun::external::re_sdk_types::ViewClassIdentifier;
use rerun::external::re_ui::{self, Help};
use rerun::external::re_viewer_context::{
    Annotations, IdentifiedViewSystem as _, IndicatedEntities, Item, ItemContext,
    MissingChunkReporter, PerVisualizerType, RecommendedVisualizers, SystemCommand,
    SystemCommandSender as _, SystemExecutionOutput, ViewClass, ViewClassLayoutPriority,
    ViewClassRegistryError, ViewClassUiOutput, ViewQuery, ViewSpawnHeuristics, ViewState,
    ViewSystemExecutionError, ViewSystemIdentifier, ViewSystemRegistrator, ViewerContext,
    VisualizableReason,
};

use crate::box_edit::Box9Dof;
use crate::slice_visualizer::{
    SliceBox, SliceBoxesOutput, SliceBoxesVisualizer, SlicePointsVisualizer,
};

/// Names new boxes uniquely for the lifetime of the process.
static NEW_BOX_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Default extent of a freshly created box, in scene units. Deliberately large
/// and cubic: it is meant to be obvious on screen and then trimmed down in the
/// slice views.
const NEW_BOX_SIZE: [f32; 3] = [5.0, 5.0, 5.0];
/// Fallback distance in front of the origin, used only when nothing has been
/// hovered in the 3D view yet.
const NEW_BOX_FORWARD: f32 = 10.0;
/// Key that drops a new box at the point currently hovered in the 3D view.
const PLACE_KEY: egui::Key = egui::Key::N;
/// Copy the selected box's shape and class.
///
/// Plain letters rather than ctrl chords: the viewer handles its own shortcuts
/// before any view runs, so ctrl-C is already spoken for (it copies the entity
/// path) and cannot be reclaimed with `consume_key`.
const COPY_KEY: egui::Key = egui::Key::C;
/// Paste the copied shape at the pointer.
const PASTE_KEY: egui::Key = egui::Key::V;
/// How far a pasted box is offset when there is no hover position to use, as a
/// multiple of its own width -- clear of the original rather than hidden inside it.
const PASTE_OFFSET: f32 = 2.5;
/// Hold this to drag the selected box around the 3D view with the pointer.
///
/// This is a held key rather than a mouse drag because the built-in 3D view
/// claims every mouse button for the camera (primary rotates, secondary pans,
/// middle rolls) with no modifier escape, and a view cannot intercept another
/// view's input. Holding a key leaves the pointer free, so the box can follow
/// the cursor without the camera moving underneath it.
const GRAB_KEY: egui::Key = egui::Key::G;
/// Lateral stagger between successive new boxes, so they don't stack exactly.
const NEW_BOX_STAGGER: f32 = 2.5;
/// Class ids are probed rather than enumerated, because `Annotations` exposes a
/// resolver but no iterator over its ontology.
const MAX_CLASS_ID_PROBE: u16 = 256;
/// Entity name used for created boxes when no existing box reveals the prefix.
const FALLBACK_BOX_PREFIX: &str = "annotations";

#[derive(Default)]
pub struct BoxListState {
    /// Cached `(class_id, label)` ontology, keyed on the annotation context's
    /// row id so it is only rebuilt when the context actually changes.
    ontology: Vec<(u16, String)>,
    ontology_key: Option<RowId>,
    /// Class assigned to the next created box.
    new_class: Option<u16>,
    /// Copied box shape and class. Position is not copied: a paste lands under
    /// the pointer, or clear of the original when there is nowhere to point.
    clipboard: Option<(Box9Dof, Option<u16>)>,
    /// Grab in progress: the box being moved and its offset from the cursor's
    /// picked position, so the box does not snap its centre to the pointer.
    grab: Option<(EntityPath, Vec3)>,
    /// Last world position hovered in a 3D view. Remembered so that moving the
    /// pointer off the 3D view (onto the button, say) does not lose the target.
    last_hover: Option<glam::Vec3>,
}

impl ViewState for BoxListState {
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

#[derive(Default)]
pub struct BoxListView;

impl ViewClass for BoxListView {
    fn identifier() -> ViewClassIdentifier {
        "BoxList".into()
    }

    fn display_name(&self) -> &'static str {
        "Boxes"
    }

    fn icon(&self) -> &'static re_ui::Icon {
        &re_ui::icons::VIEW_GENERIC
    }

    fn help(&self, _os: egui::os::OperatingSystem) -> Help {
        Help::new("Box overview").markdown(
            "Every box in the current frame.\n\n\
             - click a row to select it; the slice views follow the selection\n\
             - use the dropdown to change a box's class\n\
             - point at a spot in the 3D view and press `N` to drop a new box there\n\
             - hold `B` and drag in the 3D view to brush points; a box is fitted to them\n\
             - `C` / `V` copy a box's size and class and paste it at the pointer\n\
             - backspace deletes the selected box\n\
             - select a box, then hold `G` and move the pointer to drag it in 3D\n\
             - \"New box\" does the same at the last point you hovered",
        )
    }

    fn on_register(
        &self,
        system_registry: &mut ViewSystemRegistrator<'_>,
    ) -> Result<(), ViewClassRegistryError> {
        system_registry.register_visualizer::<SliceBoxesVisualizer>()
    }

    fn new_state(&self) -> Box<dyn ViewState> {
        Box::<BoxListState>::default()
    }

    fn layout_priority(&self) -> ViewClassLayoutPriority {
        Default::default()
    }

    fn spawn_heuristics(
        &self,
        ctx: &ViewerContext<'_>,
        include_entity: &dyn Fn(&EntityPath) -> bool,
    ) -> ViewSpawnHeuristics {
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
        // Boxes only: the point cloud would just be dead weight in a list.
        RecommendedVisualizers::default_many(
            visualizers
                .iter()
                .map(|(viz, _)| *viz)
                .filter(|viz| *viz != SlicePointsVisualizer::identifier()),
        )
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
        let boxes = system_output
            .visualizer_data_or_default::<SliceBoxesOutput>(SliceBoxesVisualizer::identifier())?;
        let boxes: &[SliceBox] = boxes.as_ref();

        let state = state
            .as_any_mut()
            .downcast_mut::<BoxListState>()
            .ok_or(ViewSystemExecutionError::StateCastError("BoxListState"))?;

        let annotations = crate::ontology::load(ctx);
        refresh_ontology(state, annotations.as_deref());
        // Cloned so the rows can borrow `ui` mutably without also holding `state`.
        let ontology = state.ontology.clone();
        if state.new_class.is_none() {
            state.new_class = ontology.first().map(|(id, _)| *id);
        }

        // The 3D view records the picked world position under the pointer every
        // frame it is hovered; that is our placement target.
        let map_from_ego = re_view_spatial_fork::frames::map_from_ego(ctx);
        if let Some(ItemContext::ThreeD { pos: Some(pos), .. }) =
            ctx.selection_state().hovered_item_context()
        {
            // The 3D view reports positions in its own (ego) frame; annotations
            // are stored in map.
            state.last_hover = Some(map_from_ego.transform_point3(*pos));
        }

        // Drag the selected box along whatever the pointer is over.
        let grab_held = ui.input(|i| i.key_down(GRAB_KEY));
        match (grab_held, state.grab.clone()) {
            (true, None) => {
                if let (Some(hover), Some(selected)) = (
                    state.last_hover,
                    ctx.selection()
                        .iter_items()
                        .filter_map(|item| item.entity_path())
                        .find_map(|p| boxes.iter().find(|b| &b.entity == p)),
                ) {
                    state.grab = Some((selected.entity.clone(), selected.bbox.center - hover));
                }
            }
            (true, Some((entity, offset))) => {
                if let (Some(hover), Some(b)) = (
                    state.last_hover,
                    boxes.iter().find(|b| b.entity == entity),
                ) {
                    let moved = Box9Dof {
                        center: hover + offset,
                        ..b.bbox
                    };
                    if moved.center != b.bbox.center {
                        write_pose(ctx, query, &entity, &moved);
                    }
                }
            }
            (false, Some(_)) => state.grab = None,
            (false, None) => {}
        }

        let box_prefix = boxes
            .first()
            .and_then(|b| b.entity.parent())
            .unwrap_or_else(|| query.space_origin.join(&FALLBACK_BOX_PREFIX.into()));

        // A finished brush stroke becomes a box here rather than in the fork,
        // which has no notion of slot naming or the active class.
        if let Some(fit) = re_view_spatial_fork::brush::take_pending() {
            create_box(
                ctx,
                query,
                &box_prefix,
                state.new_class,
                &ontology,
                Some(fit.center),
                Some((fit.half_size, fit.rotation)),
            );
        }

        // --- keyboard: copy, paste, delete ---------------------------------
        let selected_box = ctx
            .selection()
            .iter_items()
            .filter_map(|item| item.entity_path())
            .find_map(|path| boxes.iter().find(|b| &b.entity == path));

        // Detecting "is the user typing" took three attempts, so for the record:
        // `egui_wants_keyboard_input` is always true in this viewer; a focused
        // widget is not enough, because clicking the 3D view focuses it and that
        // is exactly where the shortcuts are wanted; and text events fire for any
        // printable key, focus or not, which silently killed C and V while letting
        // Backspace through. What actually distinguishes a text field is whether
        // the focused widget owns TextEdit state.
        //
        // `consume_key` also takes the event so nothing downstream sees it twice,
        // and it demands an exact modifier match -- ctrl-C stays the viewer's.
        let typing = ui
            .ctx()
            .memory(|m| m.focused())
            .is_some_and(|id| egui::TextEdit::load_state(ui.ctx(), id).is_some());
        if !typing {
            let (copy, paste, delete) = ui.input_mut(|i| {
                (
                    i.consume_key(egui::Modifiers::NONE, COPY_KEY),
                    i.consume_key(egui::Modifiers::NONE, PASTE_KEY),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace)
                        || i.consume_key(egui::Modifiers::NONE, egui::Key::Delete),
                )
            });

            if copy {
                if let Some(b) = selected_box {
                    state.clipboard = Some((b.bbox, b.class_id));
                    re_log::info!("copied {} ({:?})", b.entity, b.bbox.half_size * 2.0);
                }
            }

            if paste {
                if let Some((bbox, class_id)) = state.clipboard {
                    let center = state.last_hover.unwrap_or_else(|| {
                        bbox.center
                            + bbox.rotation
                                * Vec3::new(0.0, bbox.half_size.y * PASTE_OFFSET, 0.0)
                    });
                    create_box(
                        ctx,
                        query,
                        &box_prefix,
                        class_id.or(state.new_class),
                        &ontology,
                        Some(center),
                        Some((bbox.half_size, bbox.rotation)),
                    );
                }
            }

            if delete {
                if let Some(b) = selected_box {
                    delete_box(ctx, query, &b.entity);
                }
            }
        }

        // --- creation ------------------------------------------------------
        let place_requested = ui.input(|i| i.key_pressed(PLACE_KEY));
        if place_requested {
            re_log::info!("place key pressed; hover = {:?}", state.last_hover);
        }

        ui.horizontal(|ui| {
            let hint = match state.last_hover {
                Some(p) => format!("place at ({:.1}, {:.1}, {:.1})", p.x, p.y, p.z),
                None => format!("no 3D hover yet; will drop {NEW_BOX_FORWARD} m ahead"),
            };
            if ui.button("New box").on_hover_text(hint).clicked() || place_requested {
                create_box(
                    ctx,
                    query,
                    &box_prefix,
                    state.new_class,
                    &ontology,
                    state.last_hover,
                    None,
                );
            }

            let selected_name = state
                .new_class
                .and_then(|id| class_name(&ontology, id))
                .unwrap_or("(no class)");
            egui::ComboBox::from_id_salt("new_box_class")
                .selected_text(selected_name)
                .show_ui(ui, |ui| {
                    for (id, name) in &ontology {
                        if ui
                            .selectable_label(state.new_class == Some(*id), name)
                            .clicked()
                        {
                            state.new_class = Some(*id);
                        }
                    }
                });
        });

        let mut radius = crate::settings::point_radius();
        if ui
            .add(
                egui::Slider::new(&mut radius, 0.005..=0.4)
                    .logarithmic(true)
                    .text("point size"),
            )
            .changed()
            // Guard against a spurious change on first draw (e.g. egui clamping
            // the value into the slider's range) overwriting the logged radius.
            && (radius - crate::settings::point_radius()).abs() > 1e-6
        {
            crate::settings::set_point_radius(radius);
            set_point_radius_on_clouds(ctx, query, radius);
        }

        let clouds = point_clouds(ctx, query);
        let max_sweeps = clouds
            .iter()
            .filter_map(|(_, sweep)| *sweep)
            .max()
            .map_or(0, |k| k + 1);
        if max_sweeps > 1 {
            let mut shown = crate::settings::sweeps_shown().min(max_sweeps);
            if ui
                .add(egui::Slider::new(&mut shown, 1..=max_sweeps).text("sweeps"))
                .changed()
            {
                crate::settings::set_sweeps_shown(shown);
                set_point_radius_on_clouds(ctx, query, crate::settings::point_radius());
            }
        }

        ui.label(
            egui::RichText::new(format!(
                "3D: {} places a {}\u{d7}{}\u{d7}{} m box \u{b7} B+drag brush-fits \u{b7} \
                 C/V copy-paste \u{b7} backspace deletes",
                PLACE_KEY.name(),
                NEW_BOX_SIZE[0],
                NEW_BOX_SIZE[1],
                NEW_BOX_SIZE[2],
            ))
            .weak()
            .small(),
        );

        ui.separator();

        if boxes.is_empty() {
            ui.label("No boxes in this frame.");
            return Ok(Default::default());
        }

        // --- the list ------------------------------------------------------
        let selected: Vec<&EntityPath> = ctx
            .selection()
            .iter_items()
            .filter_map(|item| item.entity_path())
            .collect();

        ui.label(format!("{} boxes", boxes.len()));

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for b in boxes {
                    let is_selected = selected.contains(&&b.entity);
                    let name = b
                        .entity
                        .last()
                        .map(|part| part.ui_string())
                        .unwrap_or_default();
                    let class = b
                        .class_id
                        .and_then(|id| class_name(&ontology, id))
                        .unwrap_or("(unclassified)");
                    // Distance from the machine, which is what matters when
                    // scanning the list; the map origin is arbitrary.
                    let distance = (b.bbox.center - Vec3::from(map_from_ego.translation)).length();

                    ui.horizontal(|ui| {
                        let label = format!("{class}  ·  {distance:.1} m");
                        let row = ui.selectable_label(is_selected, label).on_hover_text(&name);
                        if row.clicked() {
                            ctx.command_sender()
                                .send_system(SystemCommand::set_selection(Item::InstancePath(
                                    InstancePath::entity_all(b.entity.clone()),
                                )));
                        }

                        let mut is_static =
                            re_view_spatial_fork::static_boxes::is_static(&b.entity);
                        if ui
                            .checkbox(&mut is_static, "static")
                            .on_hover_text(
                                "Show this box on every frame, for objects that do not move",
                            )
                            .changed()
                        {
                            set_static(ctx, query, b, is_static);
                        }

                        row.context_menu(|ui| {
                            if ui.button("Delete box").clicked() {
                                delete_box(ctx, query, &b.entity);
                                ui.close();
                            }
                        });

                        egui::ComboBox::from_id_salt(("class", &b.entity))
                            .selected_text("edit")
                            .width(60.0)
                            .show_ui(ui, |ui| {
                                for (id, cname) in &ontology {
                                    if ui
                                        .selectable_label(b.class_id == Some(*id), cname)
                                        .clicked()
                                    {
                                        write_class(ctx, query, &b.entity, *id, cname);
                                    }
                                }
                            });
                    });
                }
            });

        Ok(Default::default())
    }
}

/// Push a new point radius onto every point cloud under the view origin.
///
/// The slice views paint their own points and just read the shared setting, but
/// the 3D view renders through the stock `Points3D` visualizer, which takes its
/// radius from the data. Writing the component is what makes both agree.
///
/// Entities are discovered rather than hard-coded so that a cloud split across
/// several entities -- one per lidar sweep, say -- is all covered.
fn set_point_radius_on_clouds(ctx: &ViewerContext<'_>, query: &ViewQuery<'_>, radius: f32) {
    let sweeps_shown = crate::settings::sweeps_shown();
    for (path, sweep) in point_clouds(ctx, query) {
        // Sweeps beyond the slider are collapsed to zero radius rather than
        // hidden: hiding an entity means writing a blueprint override per view,
        // and a panel only knows its own view id.
        let hidden = sweep.is_some_and(|k| k >= sweeps_shown);
        let archetype = rerun::Points3D::update_fields()
            .with_radii([if hidden { 0.0 } else { radius }]);
        append(ctx, query, &path, &archetype);
    }
}

/// Entities holding point data under the view origin, with their sweep index if
/// they follow the `sweep_<k>` naming the feed uses.
///
/// Existence is tested with a batch-aware query: a point cloud is a batch of
/// thousands of positions, so the mono `latest_at_component` accessor returns
/// nothing for it and would filter every cloud out.
fn point_clouds(ctx: &ViewerContext<'_>, query: &ViewQuery<'_>) -> Vec<(EntityPath, Option<u32>)> {
    let recording = ctx.recording();
    let at = ctx.current_query();
    let positions = rerun::Points3D::descriptor_positions().component;

    recording
        .sorted_entity_paths()
        .filter(|path| path.starts_with(query.space_origin))
        .filter(|path| {
            recording
                .latest_at(&at, path, [positions])
                .component_batch_raw(positions)
                .is_some_and(|array| !array.is_empty())
        })
        .map(|path| (path.clone(), crate::settings::sweep_index(path)))
        .collect()
}

/// Rebuild the cached ontology if the annotation context changed.
fn refresh_ontology(state: &mut BoxListState, annotations: Option<&Annotations>) {
    let Some(annotations) = annotations else {
        return;
    };
    if state.ontology_key == Some(annotations.row_id()) {
        return;
    }

    state.ontology = (0..MAX_CLASS_ID_PROBE)
        .filter_map(|id| {
            let resolved =
                annotations.resolved_class_description(Some(rerun::components::ClassId::from(id)));
            resolved.class_description.map(|desc| {
                let label = desc
                    .info
                    .label
                    .as_ref()
                    .map(|l| l.to_string())
                    .unwrap_or_else(|| format!("class {id}"));
                (id, label)
            })
        })
        .collect();
    state.ontology_key = Some(annotations.row_id());
}

fn class_name(ontology: &[(u16, String)], id: u16) -> Option<&str> {
    ontology
        .iter()
        .find(|(cid, _)| *cid == id)
        .map(|(_, name)| name.as_str())
}

/// Mark a box static (present on every frame) or return it to this frame only.
///
/// Turning it on re-writes the pose with no timeline. Turning it off cannot
/// delete that row -- recordings are append-only -- so it writes an empty static
/// batch, which is how a component is cleared, and then re-writes the pose at
/// the current time so the box does not vanish.
fn set_static(ctx: &ViewerContext<'_>, query: &ViewQuery<'_>, b: &SliceBox, make_static: bool) {
    let pose = rerun::Boxes3D::from_centers_and_half_sizes(
        [(b.bbox.center.x, b.bbox.center.y, b.bbox.center.z)],
        [(
            b.bbox.half_size.x,
            b.bbox.half_size.y,
            b.bbox.half_size.z,
        )],
    )
    .with_quaternions([rerun::Quaternion::from_xyzw(b.bbox.rotation.to_array())]);

    if make_static {
        re_view_spatial_fork::static_boxes::set_static(&b.entity, true);
        append(ctx, query, &b.entity, &pose);
    } else {
        // Clear the static row first, while the registry still routes writes
        // there, then drop the flag and re-write temporally.
        append(ctx, query, &b.entity, &rerun::Boxes3D::clear_fields());
        re_view_spatial_fork::static_boxes::set_static(&b.entity, false);
        append(ctx, query, &b.entity, &pose);
    }
    re_log::info!("{} static = {make_static}", b.entity);
}

/// Delete a box by clearing its entity from the current time onward.
///
/// Recordings are append-only, so this is a tombstone rather than an erasure:
/// earlier times still hold the box. That also means deletion is per-frame,
/// which is the right behaviour for an object that leaves the scene.
fn delete_box(ctx: &ViewerContext<'_>, query: &ViewQuery<'_>, entity: &EntityPath) {
    re_log::info!("deleting {entity}");
    append(ctx, query, entity, &rerun::archetypes::Clear::recursive());
}

/// Rewrite only the class of an existing box.
///
/// Components are stored per-column, so writing just the class leaves the
/// geometry chunks untouched and latest-at still resolves them.
fn write_class(
    ctx: &ViewerContext<'_>,
    query: &ViewQuery<'_>,
    entity: &EntityPath,
    class_id: u16,
    label: &str,
) {
    let archetype = rerun::Boxes3D::update_fields()
        .with_class_ids([class_id])
        .with_labels([label]);
    append(ctx, query, entity, &archetype);
}

/// Rewrite a box's pose, leaving its class and label alone.
fn write_pose(ctx: &ViewerContext<'_>, query: &ViewQuery<'_>, entity: &EntityPath, b: &Box9Dof) {
    let archetype = rerun::Boxes3D::from_centers_and_half_sizes(
        [(b.center.x, b.center.y, b.center.z)],
        [(b.half_size.x, b.half_size.y, b.half_size.z)],
    )
    .with_quaternions([rerun::Quaternion::from_xyzw(b.rotation.to_array())]);
    append(ctx, query, entity, &archetype);
}

/// Create a new box at `position` (or ahead of the origin) and select it.
fn create_box(
    ctx: &ViewerContext<'_>,
    query: &ViewQuery<'_>,
    prefix: &EntityPath,
    class_id: Option<u16>,
    ontology: &[(u16, String)],
    position: Option<Vec3>,
    shape: Option<(Vec3, glam::Quat)>,
) -> EntityPath {
    let n = NEW_BOX_COUNTER.fetch_add(1, Ordering::Relaxed);
    let entity = prefix.join(&format!("new_{n}").into());
    re_log::info!("creating box at {entity} (hover = {position:?}, class = {class_id:?})");

    // The viewer builds its "which visualizer may draw this entity" index purely
    // from schema-addition events, additively. An entity path first seen now
    // lands in the store but is never marked visualizable, so nothing renders
    // it. Slots therefore have to be reserved up front by the logging side; see
    // `vision3d.viz.reserve_box_slots`.
    if !ctx.recording().is_logged_entity(&entity) {
        re_log::warn!(
            "{entity} was not reserved up front, so the viewer will not render it. \
             Re-run the feed with a larger --box-slots."
        );
    }

    let center = position.unwrap_or_else(|| {
        // Nothing hovered yet: drop it ahead of the origin, staggered so repeated
        // clicks don't pile boxes on the exact same spot.
        Vec3::new(
            NEW_BOX_FORWARD,
            ((n % 5) as f32 - 2.0) * NEW_BOX_STAGGER,
            0.0,
        )
    });

    // A brushed box arrives already fitted; a keyed one gets the default cube.
    let (half, rotation) = shape.unwrap_or((
        Vec3::new(
            NEW_BOX_SIZE[0] * 0.5,
            NEW_BOX_SIZE[1] * 0.5,
            NEW_BOX_SIZE[2] * 0.5,
        ),
        glam::Quat::IDENTITY,
    ));

    let mut archetype = rerun::Boxes3D::from_centers_and_half_sizes(
        [(center.x, center.y, center.z)],
        [(half.x, half.y, half.z)],
    )
    .with_quaternions([rerun::Quaternion::from_xyzw(rotation.to_array())]);

    if let Some(class_id) = class_id {
        archetype = archetype.with_class_ids([class_id]);
        if let Some(name) = class_name(ontology, class_id) {
            archetype = archetype.with_labels([name]);
        }
    }

    append(ctx, query, &entity, &archetype);

    ctx.command_sender()
        .send_system(SystemCommand::set_selection(Item::InstancePath(
            InstancePath::entity_all(entity.clone()),
        )));

    entity
}

/// Append one row for `entity` at the current time.
fn append(
    ctx: &ViewerContext<'_>,
    query: &ViewQuery<'_>,
    entity: &EntityPath,
    archetype: &dyn rerun::AsComponents,
) {
    let Some(timeline) = ctx.recording().timelines().get(&query.timeline).copied() else {
        re_log::warn_once!("no timeline named {}, edit dropped", query.timeline);
        return;
    };

    let timepoint = re_view_spatial_fork::static_boxes::timepoint_for(
        entity,
        TimePoint::from([(timeline, query.latest_at)]),
    );

    match Chunk::builder(entity.clone())
        .with_archetype_auto_row(timepoint, archetype)
        .build()
    {
        Ok(chunk) => {
            re_log::info!(
                "appending {} to {} at {:?}",
                entity,
                ctx.store_id(),
                query.latest_at
            );
            ctx.command_sender()
                .send_system(SystemCommand::AppendToStore(
                    ctx.store_id().clone(),
                    vec![chunk],
                ));
        }
        Err(err) => re_log::error_once!("failed to build box chunk: {err}"),
    }
}
