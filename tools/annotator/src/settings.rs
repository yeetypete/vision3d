//! Settings shared between the annotator's views.
//!
//! The point radius is edited in the box panel but consumed by the slice views
//! too, and views cannot see each other's state, so it lives here.

use std::sync::atomic::{AtomicU32, Ordering};

/// Point radius in scene units, as f32 bits.
static POINT_RADIUS: AtomicU32 = AtomicU32::new(0);

/// Radius the feed logs by default, used until the slider is touched.
pub const DEFAULT_POINT_RADIUS: f32 = 0.04;

pub fn point_radius() -> f32 {
    match POINT_RADIUS.load(Ordering::Relaxed) {
        0 => DEFAULT_POINT_RADIUS,
        bits => f32::from_bits(bits),
    }
}

pub fn set_point_radius(radius: f32) {
    POINT_RADIUS.store(radius.to_bits(), Ordering::Relaxed);
}

/// How many lidar sweeps to show, counting back from the newest.
static SWEEPS_SHOWN: AtomicU32 = AtomicU32::new(1);

pub fn sweeps_shown() -> u32 {
    SWEEPS_SHOWN.load(Ordering::Relaxed).max(1)
}

pub fn set_sweeps_shown(count: u32) {
    SWEEPS_SHOWN.store(count, Ordering::Relaxed);
}

/// Sweep index of an entity following the feed's `sweep_<k>` naming.
///
/// Returns `None` for clouds not split by sweep, which are always shown.
pub fn sweep_index(path: &rerun::external::re_log_types::EntityPath) -> Option<u32> {
    path.last()
        .and_then(|part| part.unescaped_str().strip_prefix("sweep_")?.parse().ok())
}

/// Whether a cloud at `path` should be drawn, given the sweeps slider.
pub fn cloud_visible(path: &rerun::external::re_log_types::EntityPath) -> bool {
    sweep_index(path).is_none_or(|k| k < sweeps_shown())
}
