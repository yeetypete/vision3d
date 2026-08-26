//! Showing and hiding a whole section of boxes, in every view at once.
//!
//! Visibility is a blueprint override rather than anything in the recording, so
//! hiding a section changes nothing about the data -- and because the override
//! is inherited down the entity tree, one write per view covers every box in
//! the section, including the ones projected into the camera panels.
//!
//! The override has to be written per view, and a view panel only knows its own
//! id, so the view ids are recovered from the blueprint's own entity paths.

use rerun::external::re_chunk::{Chunk, TimePoint};
use rerun::external::re_log;
use rerun::external::re_log_types::EntityPath;
use rerun::external::re_sdk_types::blueprint::archetypes::{EntityBehavior, ViewContents};
use rerun::external::re_viewer_context::{SystemCommand, SystemCommandSender as _, ViewerContext};

/// First path part of a view's blueprint subtree, as `ViewContents` writes it.
const VIEW_ROOT: &str = "view";

/// Show or hide `entity` and everything beneath it, in every view.
pub fn set_visible(ctx: &ViewerContext<'_>, entity: &EntityPath, visible: bool) {
    let behavior = EntityBehavior::update_fields().with_visible(visible);

    let chunks: Vec<Chunk> = view_ids(ctx)
        .into_iter()
        .filter_map(|view_id| {
            let path = ViewContents::blueprint_base_visualizer_path_for_entity(view_id, entity);
            match Chunk::builder(path)
                .with_archetype_auto_row(TimePoint::default(), &behavior)
                .build()
            {
                Ok(chunk) => Some(chunk),
                Err(err) => {
                    re_log::error_once!("failed to build a visibility override: {err}");
                    None
                }
            }
        })
        .collect();

    if chunks.is_empty() {
        re_log::warn_once!("no views to apply {entity} visibility to");
        return;
    }

    // The blueprint store, not the recording: this is a display decision.
    ctx.command_sender()
        .send_system(SystemCommand::AppendToStore(
            ctx.blueprint_db().store_id().clone(),
            chunks,
        ));
}

/// Every view in the blueprint, by id.
fn view_ids(ctx: &ViewerContext<'_>) -> Vec<uuid::Uuid> {
    let mut ids: Vec<uuid::Uuid> = ctx
        .blueprint_db()
        .sorted_entity_paths()
        .filter_map(|path| {
            let parts = path.as_slice();
            (parts.len() >= 2 && parts[0].unescaped_str() == VIEW_ROOT)
                .then(|| uuid::Uuid::parse_str(parts[1].unescaped_str()).ok())
                .flatten()
        })
        .collect();
    ids.sort();
    ids.dedup();
    ids
}
