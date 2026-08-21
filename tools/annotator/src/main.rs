//! PoC 3D bounding-box annotator built on top of the Rerun viewer.
//!
//! Layout (driven by the blueprint sent from Python, see `vision3d.viz.annotator_blueprint`):
//!
//! - centre: forked 3D view with the point cloud and 3D boxes, where dragging a
//!           box moves it (see `vendor/re_view_spatial`)
//! - bottom: built-in 2D views, one per camera, with boxes projected via the pinhole
//! - far left: a list of every box in the frame, for navigation, relabelling,
//!           and creating new boxes
//! - left:   three custom slice views (BEV / front / side) locked to the *selected box's*
//!           own coordinate frame, where the box is an axis-aligned rectangle that can be
//!           dragged, resized and rotated with the mouse.
//!
//! Launch with a recording, or with none to be asked for one:
//!
//! ```text
//! cargo run -- /path/to/recording.mcap
//! ```
//!
//! The three slice views together cover all 9 degrees of freedom:
//! translation and size from dragging the rectangle body/edges, and one rotation axis
//! per view from dragging the rotation handle.

use rerun::external::{re_crash_handler, re_grpc_server, re_log, re_memory, re_viewer, tokio};

mod box_edit;
mod box_list_view;
mod export;
mod loader;
mod ontology;
mod settings;
mod slice_view;
mod slice_visualizer;

// Keeps Rerun's memory accounting working, and mimalloc is simply faster.
#[global_allocator]
static GLOBAL: re_memory::AccountingAllocator<mimalloc::MiMalloc> =
    re_memory::AccountingAllocator::new(mimalloc::MiMalloc);

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let main_thread_token = re_viewer::MainThreadToken::i_promise_i_am_on_the_main_thread();

    re_log::setup_logging();
    re_crash_handler::install_crash_handlers(re_viewer::build_info());

    // Listen for gRPC connections from the Python SDK, exactly like the stock viewer.
    let (rx, _grpc_server_handle) = re_grpc_server::spawn_with_recv(
        "0.0.0.0:9876".parse()?,
        Default::default(),
        re_grpc_server::shutdown::never(),
    );

    let app_env = re_viewer::AppEnvironment::Custom("vision3d annotator".to_owned());
    let startup_options = re_viewer::StartupOptions::default();

    re_viewer::run_native_app(
        main_thread_token,
        Box::new(move |cc| {
            let mut app = re_viewer::App::new(
                main_thread_token,
                re_viewer::build_info(),
                app_env,
                startup_options,
                cc,
                None,
                re_viewer::AsyncRuntimeHandle::from_current_tokio_runtime_or_wasmbindgen()
                    .expect("failed to get an async runtime handle"),
            );

            // One view class per slice axis, so the blueprint can place all three
            // independently in the left column. Registration is by type, so each
            // axis is a distinct type via its marker.
            app.add_view_class::<slice_view::BoxSliceView<slice_view::AxisZ>>()
                .expect("failed to register the top slice view");
            app.add_view_class::<slice_view::BoxSliceView<slice_view::AxisY>>()
                .expect("failed to register the front slice view");
            app.add_view_class::<slice_view::BoxSliceView<slice_view::AxisX>>()
                .expect("failed to register the side slice view");

            app.add_view_class::<box_list_view::BoxListView>()
                .expect("failed to register the box list view");

            // Forked copy of the built-in 3D view, patched so dragging a box
            // moves it instead of orbiting the camera. Registered under its own
            // identifier so it sits alongside the stock view.
            app.add_view_class::<re_view_spatial_fork::SpatialView3D>()
                .expect("failed to register the annotate 3D view");

            // Tell the fork where the ego pose lives, so it can convert between
            // the view's frame and the frame annotations are stored in. Must match
            // `EGO_ENTITY` in the feed.
            re_view_spatial_fork::frames::set_ego_path("world/ego".into());

            app.add_log_receiver(rx);

            // The panel that hosts "Open bag…" is a view placed by the
            // blueprint, and the blueprint arrives with the data -- so a viewer
            // with nothing loaded has no button to press. Bootstrap it: a path on
            // the command line loads straight away, otherwise ask for one.
            match std::env::args().skip(1).find(|a| !a.starts_with('-')) {
                Some(path) => loader::load_path(path.into()),
                None => loader::pick_and_load(None),
            }

            Ok(Box::new(app))
        }),
        None,
    )?;

    Ok(())
}
