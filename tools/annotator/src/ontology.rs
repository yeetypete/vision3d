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
///
/// `AnnotationMap` finds candidate entities through a global store subscriber
/// that registers itself lazily and replays nothing, so a context logged before
/// the first call here is invisible for the rest of the session -- which is
/// silent, and looks like a recording with no classes at all. `main` registers
/// that subscriber during startup so the race cannot happen.
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

#[cfg(test)]
mod tests {
    use rerun::external::re_chunk::{Chunk, LatestAtQuery, TimePoint};
    use rerun::external::re_entity_db::EntityDb;
    use rerun::external::re_log_types::{StoreId, StoreKind};
    use rerun::external::re_viewer_context::{AnnotationContextStoreSubscriber, AnnotationMap};
    use std::sync::Arc;

    /// The annotation context is found when the subscriber exists beforehand.
    ///
    /// `AnnotationMap` reads the entities that carry a context from a global
    /// store subscriber that registers lazily and does not replay, so a context
    /// logged before the first lookup is invisible for the rest of the session.
    /// The annotator registers the subscriber at startup for exactly this
    /// reason; this pins the property that makes doing so sufficient.
    #[test]
    fn context_is_visible_to_a_subscriber_registered_before_the_store() {
        AnnotationContextStoreSubscriber::subscription_handle();

        let mut db = EntityDb::new(StoreId::random(StoreKind::Recording, "test"));
        let entity = "world/annotations";
        let context =
            rerun::AnnotationContext::new([(0, "truck"), (1, "truck_cabin"), (2, "truck_bed")]);
        let chunk = Chunk::builder(entity)
            .with_archetype_auto_row(TimePoint::STATIC, &context)
            .build()
            .expect("failed to build the annotation context chunk");
        db.add_chunk(&Arc::new(chunk))
            .expect("failed to add the chunk");

        let mut map = AnnotationMap::default();
        map.load(&db, &LatestAtQuery::latest("time".into()));

        let annotations = map
            .0
            .into_values()
            .next()
            .expect("no annotation context found");
        for (id, label) in [(0, "truck"), (1, "truck_cabin"), (2, "truck_bed")] {
            let resolved = annotations
                .resolved_class_description(Some(rerun::components::ClassId::from(id as u16)));
            let name = resolved
                .class_description
                .and_then(|d| d.info.label.as_ref().map(|l| l.to_string()));
            assert_eq!(
                name,
                Some(label.to_owned()),
                "class {id} missing from the ontology"
            );
        }
    }
    /// A cleared entity resolves to nothing, which is what keeps the box
    /// panel's reserved slots out of its list.
    ///
    /// The annotator pre-registers 64 entity paths by logging a throwaway box
    /// and clearing it -- the schema registration is what the viewer's
    /// visualizability index needs, and the clear removes the data. The panel
    /// lists boxes with `latest_at`, so it only shows real ones as long as
    /// `latest_at` honours those tombstones.
    #[test]
    fn a_cleared_entity_resolves_to_nothing() {
        use rerun::external::re_chunk::{Chunk, LatestAtQuery, TimePoint};
        use rerun::external::re_entity_db::EntityDb;
        use rerun::external::re_log_types::{StoreId, StoreKind};
        use std::sync::Arc;

        let mut db = EntityDb::new(StoreId::random(StoreKind::Recording, "probe"));
        let path: rerun::EntityPath = "world/annotations/manual/new_0".into();

        let boxes =
            rerun::Boxes3D::from_centers_and_half_sizes([(0.0, 0.0, 0.0)], [(1.0, 1.0, 1.0)]);
        db.add_chunk(&Arc::new(
            Chunk::builder(path.clone())
                .with_archetype_auto_row(TimePoint::STATIC, &boxes)
                .build()
                .unwrap(),
        ))
        .unwrap();
        db.add_chunk(&Arc::new(
            Chunk::builder(path.clone())
                .with_archetype_auto_row(TimePoint::STATIC, &rerun::archetypes::Clear::recursive())
                .build()
                .unwrap(),
        ))
        .unwrap();

        let half = rerun::Boxes3D::descriptor_half_sizes().component;
        let got = db
            .latest_at(&LatestAtQuery::latest("time".into()), &path, [half])
            .component_batch_raw(half);
        assert!(
            got.is_none(),
            "a cleared box still resolved; the panel would list all 64 \
             reserved slots as real annotations"
        );
    }
}
