//! Which annotated boxes are marked static, and therefore where edits go.
//!
//! A static box is logged with no timeline, so it shows on every frame. Rerun
//! gives static data precedence over temporal data for the same component,
//! which sets a trap: writing an edit at the current time would be silently
//! ignored. Every write path therefore routes its timepoint through
//! [`timepoint_for`].
//!
//! This is annotator policy living in a viewer crate, which is not where it
//! belongs -- but the fork's drag handler has to honour it, and the fork cannot
//! depend on the application. One source of truth beats two.

use parking_lot::Mutex;
use re_log_types::EntityPath;

/// Small by nature -- a handful of static objects per scene -- so a Vec with a
/// linear scan avoids needing a non-const `HashSet` constructor.
static STATIC_BOXES: Mutex<Vec<EntityPath>> = Mutex::new(Vec::new());

pub fn is_static(path: &EntityPath) -> bool {
    STATIC_BOXES.lock().iter().any(|p| p == path)
}

pub fn set_static(path: &EntityPath, make_static: bool) {
    let mut boxes = STATIC_BOXES.lock();
    match (make_static, boxes.iter().position(|p| p == path)) {
        (true, None) => boxes.push(path.clone()),
        (false, Some(index)) => {
            boxes.swap_remove(index);
        }
        _ => {}
    }
}

/// The timepoint an edit to `path` must be written at.
pub fn timepoint_for(path: &EntityPath, temporal: re_chunk::TimePoint) -> re_chunk::TimePoint {
    if is_static(path) {
        re_chunk::TimePoint::STATIC
    } else {
        temporal
    }
}
