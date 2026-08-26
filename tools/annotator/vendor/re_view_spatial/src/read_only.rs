//! Which annotated boxes may not be edited.
//!
//! The recording's own boxes are shown for reference and are never written back
//! to, so every edit path has to refuse them -- otherwise a drag silently
//! produces a change with nowhere to go.
//!
//! Like [`crate::static_boxes`], this is annotator policy living in a viewer
//! crate, which is not where it belongs. The fork's drag handler has to honour
//! it and cannot depend on the application, and one source of truth beats two.

use parking_lot::Mutex;
use re_log_types::EntityPath;

/// Entity-path segment holding the read-only boxes, set once by the app so both
/// sides agree on the name.
static SECTION: Mutex<Option<String>> = Mutex::new(None);

/// Name the section whose boxes are read-only, e.g. `"source"`.
pub fn set_section(name: impl Into<String>) {
    *SECTION.lock() = Some(name.into());
}

/// Whether `path` is a box in that section.
///
/// Box paths are `<root>/<section>/<track>`, so the section is the parent.
pub fn is_read_only(path: &EntityPath) -> bool {
    let section = SECTION.lock();
    let Some(section) = section.as_deref() else {
        return false;
    };
    path.parent()
        .and_then(|parent| parent.last().cloned())
        .is_some_and(|part| part.unescaped_str() == section)
}

