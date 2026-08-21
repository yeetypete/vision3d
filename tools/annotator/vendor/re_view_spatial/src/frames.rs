//! The one transform the annotator needs: map ↔ ego.
//!
//! Sensor data is logged in the ego (lidar) frame under an ego node that carries
//! the per-frame `map_from_lidar` pose, and the 3D view is rooted at that node.
//! Points therefore render in their raw coordinates, so the view rides with the
//! machine exactly as raw sensor data would.
//!
//! Annotations, by contrast, are stored in the map frame, so an object that
//! never moved keeps one pose for the whole sequence.
//!
//! That leaves precisely two frames in play rather than an arbitrary tree, so a
//! single transform covers every conversion: picking and rays come out of the
//! view in ego coordinates, box poses come out of the store in map coordinates.

use glam::{Affine3A, Mat3, Vec3};
use parking_lot::Mutex;
use re_log_types::EntityPath;
use re_sdk_types::archetypes::Transform3D;
use re_sdk_types::components::{TransformMat3x3, Translation3D};
use re_viewer_context::ViewerContext;

/// Entity carrying the ego pose. Set by the application, whose logging side
/// chose the path.
static EGO_PATH: Mutex<Option<EntityPath>> = Mutex::new(None);

pub fn set_ego_path(path: EntityPath) {
    *EGO_PATH.lock() = Some(path);
}

/// `map_from_ego` at the current time, or identity if no ego pose is logged.
///
/// Identity is the right fallback: without an ego pose the two frames coincide,
/// which is exactly how a recording with no pose behaves.
pub fn map_from_ego(ctx: &ViewerContext<'_>) -> Affine3A {
    let Some(path) = EGO_PATH.lock().clone() else {
        return Affine3A::IDENTITY;
    };

    let query = ctx.current_query();
    let db = ctx.recording();

    let translation = db
        .latest_at_component::<Translation3D>(
            &path,
            &query,
            Transform3D::descriptor_translation().component,
        )
        .map_or(Vec3::ZERO, |(_, t)| Vec3::from_array(t.0.0));

    // Rerun stores matrices column-major, as glam does.
    let rotation = db
        .latest_at_component::<TransformMat3x3>(
            &path,
            &query,
            Transform3D::descriptor_mat3x3().component,
        )
        .map_or(Mat3::IDENTITY, |(_, m)| Mat3::from_cols_array(&m.0.0));

    Affine3A::from_mat3_translation(rotation, translation)
}

/// Move a rigid box pose between frames. Extents are unaffected.
pub fn transform_pose(
    transform: Affine3A,
    center: Vec3,
    rotation: glam::Quat,
) -> (Vec3, glam::Quat) {
    let moved_rotation = glam::Quat::from_mat3(&Mat3::from(transform.matrix3)) * rotation;
    (transform.transform_point3(center), moved_rotation.normalize())
}
