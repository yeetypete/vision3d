//! Which instants a box's pose was actually authored at.
//!
//! A box written only where it was touched holds its last pose until the next
//! write, so an object that moves jumps between the frames you edited -- and on
//! export, where each entity lives for one keyframe interval, it blinks out
//! entirely in between. Recording the authored instants lets the poses between
//! them be filled in.
//!
//! Only the marking lives here. Filling is the application's job, because it
//! needs to know which instants the recording actually has frames at; the fork
//! reports what changed and the panel acts on it next frame.
//!
//! Like [`crate::static_boxes`], this is annotator policy sitting in a viewer
//! crate so that the fork's drag handler and the application share one source
//! of truth.

use parking_lot::Mutex;
use re_log_types::EntityPath;

/// Authored instants per box, and the boxes whose track needs rebuilding.
static AUTHORED: Mutex<Vec<(EntityPath, Vec<i64>)>> = Mutex::new(Vec::new());
static DIRTY: Mutex<Vec<EntityPath>> = Mutex::new(Vec::new());
/// Trimmed lifespan per box: `(start, end)`, either open.
static BOUNDS: Mutex<Vec<(EntityPath, Option<i64>, Option<i64>)>> = Mutex::new(Vec::new());

/// Record that `path` had its pose authored at `time_ns`.
///
/// Call this from every path that writes a pose at a point in time. Writing a
/// static pose is not a keyframe: a static box has one pose for the whole
/// recording and nothing to interpolate.
pub fn mark(path: &EntityPath, time_ns: i64) {
    let mut authored = AUTHORED.lock();
    match authored.iter_mut().find(|(p, _)| p == path) {
        Some((_, times)) => {
            if let Err(index) = times.binary_search(&time_ns) {
                times.insert(index, time_ns);
            }
        }
        None => authored.push((path.clone(), vec![time_ns])),
    }
    drop(authored);

    touch(path);
}

/// Note that a box's track needs rebuilding.
pub fn touch(path: &EntityPath) {
    let mut dirty = DIRTY.lock();
    if !dirty.iter().any(|p| p == path) {
        dirty.push(path.clone());
    }
}

/// Trim the front of a box's life, or clear that trim with `None`.
pub fn set_start(path: &EntityPath, start: Option<i64>) {
    let mut bounds = BOUNDS.lock();
    match bounds.iter_mut().find(|(p, _, _)| p == path) {
        Some(entry) => entry.1 = start,
        None => bounds.push((path.clone(), start, None)),
    }
    drop(bounds);
    touch(path);
}

/// Trim the back of a box's life, or clear that trim with `None`.
pub fn set_end(path: &EntityPath, end: Option<i64>) {
    let mut bounds = BOUNDS.lock();
    match bounds.iter_mut().find(|(p, _, _)| p == path) {
        Some(entry) => entry.2 = end,
        None => bounds.push((path.clone(), None, end)),
    }
    drop(bounds);
    touch(path);
}

/// The box's trimmed lifespan, either end open when untrimmed.
pub fn bounds(path: &EntityPath) -> (Option<i64>, Option<i64>) {
    BOUNDS
        .lock()
        .iter()
        .find(|(p, _, _)| p == path)
        .map_or((None, None), |(_, s, e)| (*s, *e))
}

/// The authored instants either side of `time_ns`, exclusive.
///
/// Returns `(previous, next)`; either is `None` when this is the first or last
/// keyframe, in which case there is nothing on that side to interpolate to.
pub fn neighbours(path: &EntityPath, time_ns: i64) -> (Option<i64>, Option<i64>) {
    let authored = AUTHORED.lock();
    let Some((_, times)) = authored.iter().find(|(p, _)| p == path) else {
        return (None, None);
    };
    (
        times.iter().copied().filter(|t| *t < time_ns).next_back(),
        times.iter().copied().find(|t| *t > time_ns),
    )
}

/// Drop a single authored instant.
///
/// The pose written there is left alone: the caller refills the span, which
/// overwrites it with the interpolation across the gap. Deleting the first or
/// last keyframe therefore shortens the authored span but leaves the pose that
/// was there, since there is nothing on the far side to interpolate towards.
///
/// Returns whether there was one to drop.
pub fn remove(path: &EntityPath, time_ns: i64) -> bool {
    let mut authored = AUTHORED.lock();
    let Some((_, times)) = authored.iter_mut().find(|(p, _)| p == path) else {
        return false;
    };
    match times.binary_search(&time_ns) {
        Ok(index) => {
            times.remove(index);
            true
        }
        Err(_) => false,
    }
}

/// The authored instants of this box, in order.
pub fn times(path: &EntityPath) -> Vec<i64> {
    AUTHORED
        .lock()
        .iter()
        .find(|(p, _)| p == path)
        .map_or_else(Vec::new, |(_, times)| times.clone())
}

/// How many instants this box has been authored at.
pub fn count(path: &EntityPath) -> usize {
    AUTHORED
        .lock()
        .iter()
        .find(|(p, _)| p == path)
        .map_or(0, |(_, times)| times.len())
}

/// Forget a box's keyframes, so its next edit starts a fresh span.
pub fn forget(path: &EntityPath) {
    AUTHORED.lock().retain(|(p, _)| p != path);
    DIRTY.lock().retain(|p| p != path);
}

/// Take the boxes whose tracks need rebuilding.
pub fn take_dirty() -> Vec<EntityPath> {
    std::mem::take(&mut *DIRTY.lock())
}
