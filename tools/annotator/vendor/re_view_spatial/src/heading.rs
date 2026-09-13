//! The arrow showing which way a box faces.
//!
//! Written wherever a pose is written, never on a schedule: an arrow kept in
//! step by a per-frame comparison meant the panel wrote to the store on every
//! frame, which invalidates the query caches and makes every unrelated read
//! recompute.
//!
//! Lives in the fork so the 3D drag and the application build the same arrow.

use glam::{Quat, Vec3};
use re_log_types::EntityPath;
use re_sdk_types::archetypes::Arrows3D;

/// Entity segment carrying a box's heading arrow, beneath the box itself.
pub const SECTION: &str = "heading";

/// The entity a box's heading arrow lives at.
pub fn path_for(box_path: &EntityPath) -> EntityPath {
    box_path.join(&SECTION.into())
}

/// Build the arrow for a pose.
///
/// Sized from the box's own front face, matching what
/// `vision3d.viz.log_boxes_3d` draws, so an annotated box and a logged dataset
/// look the same.
pub fn archetype(center: Vec3, half: Vec3, rotation: Quat) -> Arrows3D {
    let forward = rotation * Vec3::X;
    let face_scale = ((2.0 * half.y) * (2.0 * half.z)).sqrt();
    let origin = center + forward * half.x;
    let vector = forward * face_scale * 0.6;

    Arrows3D::from_vectors([(vector.x, vector.y, vector.z)])
        .with_origins([(origin.x, origin.y, origin.z)])
        .with_radii([face_scale * 0.06])
        .with_colors([(255, 255, 255)])
}
