//! 9-DoF box representation and the drag math behind the slice views.
//!
//! Deliberately free of `egui`/`rerun` types so the geometry can be unit-tested
//! on its own.

use glam::{Quat, Vec3};

/// Smallest half-extent a box may be dragged down to, in scene units.
pub const MIN_HALF_SIZE: f32 = 0.025;

/// A 9-DoF oriented bounding box: centre, half-extents, and orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box9Dof {
    pub center: Vec3,
    pub half_size: Vec3,
    pub rotation: Quat,
}

impl Box9Dof {
    /// The frame this box defines: its centre and orientation.
    pub fn frame(&self) -> Frame {
        Frame {
            origin: self.center,
            rotation: self.rotation,
        }
    }
}

/// A rigid frame used to express points in some box's local coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub origin: Vec3,
    pub rotation: Quat,
}

impl Frame {
    /// World -> frame-local.
    pub fn to_local(&self, p: Vec3) -> Vec3 {
        self.rotation.inverse() * (p - self.origin)
    }

    /// Frame-local -> world.
    pub fn to_world(&self, p: Vec3) -> Vec3 {
        self.origin + self.rotation * p
    }
}

/// Which axis a slice view looks along, and therefore which plane it edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceAxis {
    /// Looks down local -Z. Edits local XY and yaw. The bird's-eye view.
    Z,
    /// Looks along local -Y. Edits local XZ and pitch.
    Y,
    /// Looks along local -X. Edits local YZ and roll.
    X,
}

impl SliceAxis {
    /// Stable identifier, matched by `SLICE_VIEW_CLASSES` on the Python side.
    pub fn identifier(self) -> &'static str {
        match self {
            Self::Z => "BoxSliceZ",
            Self::Y => "BoxSliceY",
            Self::X => "BoxSliceX",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Z => "Top (yaw)",
            Self::Y => "Front (pitch)",
            Self::X => "Side (roll)",
        }
    }

    /// Local axis indices as `(horizontal, vertical, view)`.
    ///
    /// The view axis is the one being looked along, and is also the axis the
    /// rotation handle in this view spins the box about.
    pub fn axes(self) -> (usize, usize, usize) {
        match self {
            Self::Z => (0, 1, 2),
            Self::Y => (0, 2, 1),
            Self::X => (1, 2, 0),
        }
    }
}

/// Which part of the on-screen rectangle a drag grabbed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragKind {
    /// Interior: translates the box within the view plane.
    Body,
    /// One of the four edges: moves that face, changing size and centre.
    Edge(Edge),
    /// The handle above the rectangle: spins the box about the view axis.
    Rotate,
}

/// One of the four in-plane faces of the box, in view-local terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    MinU,
    MaxU,
    MinV,
    MaxV,
}

/// Where the rotation handle sits, as a multiple of the vertical half-extent.
pub const ROTATE_HANDLE_OFFSET: f32 = 1.35;

/// Hit-test a pointer position given in view-plane local coordinates.
///
/// Hit-testing only ever runs at the start of a drag, when the box and the view
/// frame still coincide, so the rectangle is axis-aligned here.
///
/// Args are in scene units: `(u, v)` is the pointer, `(hu, hv)` the in-plane
/// half-extents, `tol` the grab tolerance (a pixel radius converted to scene
/// units by the caller).
pub fn hit_test(u: f32, v: f32, hu: f32, hv: f32, tol: f32) -> Option<DragKind> {
    let handle_v = hv * ROTATE_HANDLE_OFFSET;
    if (u.abs() <= tol * 2.0) && (v - handle_v).abs() <= tol * 2.0 {
        return Some(DragKind::Rotate);
    }

    let near_u_span = v.abs() <= hv + tol;
    let near_v_span = u.abs() <= hu + tol;

    // Edges take priority over the body so thin boxes stay resizable.
    if near_u_span && (u - hu).abs() <= tol {
        return Some(DragKind::Edge(Edge::MaxU));
    }
    if near_u_span && (u + hu).abs() <= tol {
        return Some(DragKind::Edge(Edge::MinU));
    }
    if near_v_span && (v - hv).abs() <= tol {
        return Some(DragKind::Edge(Edge::MaxV));
    }
    if near_v_span && (v + hv).abs() <= tol {
        return Some(DragKind::Edge(Edge::MinV));
    }

    (u.abs() <= hu && v.abs() <= hv).then_some(DragKind::Body)
}

/// Apply a drag to the box as it was when the drag started.
///
/// Deltas are always measured from the drag origin rather than accumulated
/// per-frame, so a drag cannot drift and releasing/re-grabbing is exact.
///
/// `du`/`dv` are the in-plane pointer delta in scene units; `angle` is the
/// rotation delta in radians and is only read for [`DragKind::Rotate`].
pub fn apply_drag(
    start: &Box9Dof,
    axis: SliceAxis,
    kind: DragKind,
    du: f32,
    dv: f32,
    angle: f32,
) -> Box9Dof {
    let (iu, iv, in_) = axis.axes();
    let mut out = *start;

    match kind {
        DragKind::Body => {
            let mut local = Vec3::ZERO;
            local[iu] = du;
            local[iv] = dv;
            out.center = start.center + start.rotation * local;
        }

        DragKind::Edge(edge) => {
            let (idx, delta, outward) = match edge {
                Edge::MaxU => (iu, du, 1.0),
                Edge::MinU => (iu, du, -1.0),
                Edge::MaxV => (iv, dv, 1.0),
                Edge::MinV => (iv, dv, -1.0),
            };

            // Moving a face by `delta` changes the half-extent by `delta/2` and
            // shifts the centre by the same amount, so the opposite face stays
            // put. Deriving the shift from the *clamped* half-extent keeps that
            // true even when the box bottoms out at MIN_HALF_SIZE.
            let old_half = start.half_size[idx];
            let new_half = (old_half + outward * delta * 0.5).max(MIN_HALF_SIZE);
            out.half_size[idx] = new_half;

            let mut shift = Vec3::ZERO;
            shift[idx] = outward * (new_half - old_half);
            out.center = start.center + start.rotation * shift;
        }

        DragKind::Rotate => {
            let mut local_axis = Vec3::ZERO;
            local_axis[in_] = 1.0;
            out.rotation = (start.rotation * Quat::from_axis_angle(local_axis, angle)).normalize();
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box() -> Box9Dof {
        Box9Dof {
            center: Vec3::new(1.0, 2.0, 3.0),
            half_size: Vec3::new(2.0, 1.0, 0.5),
            rotation: Quat::IDENTITY,
        }
    }

    #[test]
    fn body_drag_translates_in_plane_only() {
        let b = apply_drag(&unit_box(), SliceAxis::Z, DragKind::Body, 0.5, -0.25, 0.0);
        assert_eq!(b.center, Vec3::new(1.5, 1.75, 3.0));
        assert_eq!(b.half_size, unit_box().half_size);
    }

    #[test]
    fn edge_drag_holds_the_opposite_face_still() {
        let start = unit_box();
        let opposite_before = start.center.x - start.half_size.x;

        let b = apply_drag(&start, SliceAxis::Z, DragKind::Edge(Edge::MaxU), 1.0, 0.0, 0.0);

        assert!((b.half_size.x - 2.5).abs() < 1e-6);
        assert!(((b.center.x - b.half_size.x) - opposite_before).abs() < 1e-6);
    }

    #[test]
    fn edge_drag_clamps_without_moving_the_opposite_face() {
        let start = unit_box();
        let opposite_before = start.center.x - start.half_size.x;

        // Drag the +u face far past the -u face.
        let b = apply_drag(
            &start,
            SliceAxis::Z,
            DragKind::Edge(Edge::MaxU),
            -100.0,
            0.0,
            0.0,
        );

        assert!((b.half_size.x - MIN_HALF_SIZE).abs() < 1e-6);
        assert!(((b.center.x - b.half_size.x) - opposite_before).abs() < 1e-6);
    }

    #[test]
    fn rotation_handles_are_per_view_axis() {
        let start = unit_box();
        let angle = 0.3;

        for (axis, expected) in [
            (SliceAxis::Z, Vec3::Z),
            (SliceAxis::Y, Vec3::Y),
            (SliceAxis::X, Vec3::X),
        ] {
            let b = apply_drag(&start, axis, DragKind::Rotate, 0.0, 0.0, angle);
            let want = Quat::from_axis_angle(expected, angle);
            assert!(
                b.rotation.dot(want).abs() > 0.999_9,
                "{axis:?} should rotate about {expected:?}"
            );
        }
    }

    #[test]
    fn drags_are_absolute_not_incremental() {
        let start = unit_box();
        let once = apply_drag(&start, SliceAxis::Z, DragKind::Body, 1.0, 0.0, 0.0);
        let twice = apply_drag(&once, SliceAxis::Z, DragKind::Body, 1.0, 0.0, 0.0);
        // Re-applying against the *drag start* is what the view does; applying
        // against the previous result would double the motion.
        assert_ne!(once.center, twice.center);
        assert_eq!(
            apply_drag(&start, SliceAxis::Z, DragKind::Body, 1.0, 0.0, 0.0).center,
            once.center
        );
    }

    #[test]
    fn hit_test_prefers_edges_over_body() {
        assert_eq!(hit_test(0.0, 0.0, 2.0, 1.0, 0.1), Some(DragKind::Body));
        assert_eq!(
            hit_test(2.0, 0.0, 2.0, 1.0, 0.1),
            Some(DragKind::Edge(Edge::MaxU))
        );
        assert_eq!(
            hit_test(0.0, -1.0, 2.0, 1.0, 0.1),
            Some(DragKind::Edge(Edge::MinV))
        );
        assert_eq!(
            hit_test(0.0, 1.0 * ROTATE_HANDLE_OFFSET, 2.0, 1.0, 0.1),
            Some(DragKind::Rotate)
        );
        assert_eq!(hit_test(9.0, 9.0, 2.0, 1.0, 0.1), None);
    }
}
