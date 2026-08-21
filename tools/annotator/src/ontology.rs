//! Shared access to the recording's class ontology.
//!
//! Both the box panel (label dropdown) and the slice views (outline colour) need
//! the annotation context, and views cannot see each other's state, so the
//! lookup lives here.

use std::sync::Arc;

use rerun::external::egui;
use rerun::external::re_viewer_context::{AnnotationMap, Annotations, ViewerContext};

/// The annotation context covering the annotated entities.
///
/// Takes the first context in the recording, which is what the vision3d logger
/// produces (a single ontology on the box prefix). A recording with several
/// contexts would need the nearest-ancestor walk instead.
pub fn load(ctx: &ViewerContext<'_>) -> Option<Arc<Annotations>> {
    let mut map = AnnotationMap::default();
    map.load(ctx.recording(), &ctx.current_query());
    map.0.into_values().next()
}

/// The colour the 3D view draws a class in.
///
/// Resolves through the same path the built-in visualizers use, so an explicit
/// `AnnotationContext` colour and Rerun's automatic per-class colour both come
/// out matching what the 3D view shows.
pub fn class_color(ctx: &ViewerContext<'_>, class_id: Option<u16>) -> Option<egui::Color32> {
    let class_id = class_id?;
    // Fall back to an empty context rather than giving up: Rerun derives a
    // colour from the class id alone when no `AnnotationContext` is logged, and
    // the 3D view shows that colour. Bailing out here would leave 2D on the UI
    // accent while 3D showed a per-class colour.
    let annotations = load(ctx).unwrap_or_else(Annotations::missing_arc);
    annotations
        .resolved_class_description(Some(rerun::components::ClassId::from(class_id)))
        .annotation_info()
        .color()
}
