//! Visualizers that pull the point cloud and the editable boxes out of the
//! chunk store for the slice views.
//!
//! These do no rendering: the slice views paint with `egui` directly, so all
//! these do is hand over plain `glam` data.

use glam::{Quat, Vec3};
use rerun::external::egui;
use rerun::external::re_log_types::EntityPath;
use rerun::external::re_view::{DataResultQuery as _, VisualizerInstructionQueryResults};
use rerun::external::re_viewer_context::{
    AppOptions, IdentifiedViewSystem, SingleRequiredComponentConstraint, ViewContext,
    ViewContextCollection, ViewQuery, ViewSystemExecutionError, ViewSystemIdentifier,
    VisualizerExecutionOutput, VisualizerQueryInfo, VisualizerSystem,
};

use crate::box_edit::Box9Dof;

/// Points of one entity, in the frame the view is rooted at.
#[derive(Clone)]
pub struct SlicePoints {
    pub entity: EntityPath,
    pub positions: Vec<Vec3>,
    pub colors: Vec<egui::Color32>,
}

pub type SlicePointsOutput = Vec<SlicePoints>;

/// A single editable box, tagged with the entity it lives at so an edit can be
/// written straight back to the same path.
#[derive(Clone)]
pub struct SliceBox {
    pub entity: EntityPath,
    pub bbox: Box9Dof,
    pub class_id: Option<u16>,
}

pub type SliceBoxesOutput = Vec<SliceBox>;

/// Reads `Points3D` positions and colors.
#[derive(Default)]
pub struct SlicePointsVisualizer;

impl IdentifiedViewSystem for SlicePointsVisualizer {
    fn identifier() -> ViewSystemIdentifier {
        "SlicePointsVisualizer".into()
    }
}

impl VisualizerSystem for SlicePointsVisualizer {
    fn visualizer_query_info(&self, _app_options: &AppOptions) -> VisualizerQueryInfo {
        VisualizerQueryInfo {
            relevant_archetype: None,
            constraints: SingleRequiredComponentConstraint::new::<rerun::components::Position3D>(
                &rerun::Points3D::descriptor_positions(),
            )
            .into(),
            queried: [
                rerun::Points3D::descriptor_positions(),
                rerun::Points3D::descriptor_colors(),
            ]
            .into_iter()
            .collect(),
        }
    }

    fn execute(
        &self,
        ctx: &ViewContext<'_>,
        query: &ViewQuery<'_>,
        _context_systems: &ViewContextCollection,
    ) -> Result<VisualizerExecutionOutput, ViewSystemExecutionError> {
        let output = VisualizerExecutionOutput::default();
        let mut clouds: SlicePointsOutput = Vec::new();

        for (data_result, instruction) in query.iter_visualizer_instruction_for(Self::identifier())
        {
            let results = data_result.query_components_with_history(
                ctx,
                query,
                [
                    rerun::Points3D::descriptor_positions().component,
                    rerun::Points3D::descriptor_colors().component,
                ],
                instruction,
            );
            let results = VisualizerInstructionQueryResults::new(instruction, &results, &output);

            let positions_results =
                results.iter_optional(rerun::Points3D::descriptor_positions().component);
            let colors_results =
                results.iter_optional(rerun::Points3D::descriptor_colors().component);
            let positions = positions_results.slice::<[f32; 3]>();
            let mut colors_per_time = colors_results.slice::<u32>();

            for ((_time, _row_id), xyz) in positions {
                let colors = colors_per_time.next().map(|(_, c)| c);

                let mut cloud = SlicePoints {
                    entity: data_result.entity_path.clone(),
                    positions: Vec::with_capacity(xyz.len()),
                    colors: Vec::new(),
                };

                for p in xyz {
                    cloud.positions.push(Vec3::from_array(*p));
                }
                if let Some(colors) = colors {
                    cloud.colors.reserve(colors.len());
                    for c in colors {
                        let [r, g, b, _] = rerun::Color::from_u32(*c).to_array();
                        cloud.colors.push(egui::Color32::from_rgb(r, g, b));
                    }
                }

                if !cloud.positions.is_empty() {
                    clouds.push(cloud);
                }
            }
        }

        Ok(output.with_visualizer_data(clouds))
    }
}

/// Reads `Boxes3D` centres, half-extents, and orientations.
#[derive(Default)]
pub struct SliceBoxesVisualizer;

impl IdentifiedViewSystem for SliceBoxesVisualizer {
    fn identifier() -> ViewSystemIdentifier {
        "SliceBoxesVisualizer".into()
    }
}

impl VisualizerSystem for SliceBoxesVisualizer {
    fn visualizer_query_info(&self, _app_options: &AppOptions) -> VisualizerQueryInfo {
        VisualizerQueryInfo {
            relevant_archetype: None,
            constraints: SingleRequiredComponentConstraint::new::<rerun::components::HalfSize3D>(
                &rerun::Boxes3D::descriptor_half_sizes(),
            )
            .into(),
            queried: [
                rerun::Boxes3D::descriptor_half_sizes(),
                rerun::Boxes3D::descriptor_centers(),
                rerun::Boxes3D::descriptor_quaternions(),
                rerun::Boxes3D::descriptor_class_ids(),
            ]
            .into_iter()
            .collect(),
        }
    }

    fn execute(
        &self,
        ctx: &ViewContext<'_>,
        query: &ViewQuery<'_>,
        _context_systems: &ViewContextCollection,
    ) -> Result<VisualizerExecutionOutput, ViewSystemExecutionError> {
        let output = VisualizerExecutionOutput::default();
        let mut boxes: SliceBoxesOutput = Vec::new();

        for (data_result, instruction) in query.iter_visualizer_instruction_for(Self::identifier())
        {
            let results = data_result.query_components_with_history(
                ctx,
                query,
                [
                    rerun::Boxes3D::descriptor_half_sizes().component,
                    rerun::Boxes3D::descriptor_centers().component,
                    rerun::Boxes3D::descriptor_quaternions().component,
                    rerun::Boxes3D::descriptor_class_ids().component,
                ],
                instruction,
            );
            let results = VisualizerInstructionQueryResults::new(instruction, &results, &output);

            let half_size_results =
                results.iter_optional(rerun::Boxes3D::descriptor_half_sizes().component);
            let center_results =
                results.iter_optional(rerun::Boxes3D::descriptor_centers().component);
            let quat_results =
                results.iter_optional(rerun::Boxes3D::descriptor_quaternions().component);
            let class_results =
                results.iter_optional(rerun::Boxes3D::descriptor_class_ids().component);
            let half_sizes = half_size_results.slice::<[f32; 3]>();
            let mut centers = center_results.slice::<[f32; 3]>();
            let mut quats = quat_results.slice::<[f32; 4]>();
            let mut class_ids = class_results.slice::<u16>();

            for ((_time, _row_id), half) in half_sizes {
                let center = centers.next().map(|(_, c)| c).unwrap_or(&[]);
                let quat = quats.next().map(|(_, q)| q).unwrap_or(&[]);
                let class = class_ids.next().map(|(_, c)| c).unwrap_or(&[]);

                // The annotator logs one box per entity, so only the first
                // instance is editable here. Extra instances are ignored rather
                // than silently rewritten.
                let Some(half) = half.first() else {
                    continue;
                };

                let center = center.first().map_or(Vec3::ZERO, |c| Vec3::from_array(*c));
                let rotation = quat
                    .first()
                    .map_or(Quat::IDENTITY, |q| Quat::from_xyzw(q[0], q[1], q[2], q[3]));

                boxes.push(SliceBox {
                    entity: data_result.entity_path.clone(),
                    class_id: class.first().copied(),
                    bbox: Box9Dof {
                        center,
                        half_size: Vec3::from_array(*half),
                        rotation,
                    },
                });
            }
        }

        Ok(output.with_visualizer_data(boxes))
    }
}
