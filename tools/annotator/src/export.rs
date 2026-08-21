//! Writing annotations to a sidecar file, and finding where to put it.
//!
//! The store holds every edit ever made, so an export has to collapse them:
//! dragging a box writes a row per frame of the gesture, and only the last one
//! at each timestamp is the label. Chunks are read straight out of the store
//! rather than through a range query, since we want everything ever written for
//! these entities rather than a window around the cursor.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use rerun::external::parking_lot::Mutex;
use rerun::external::re_log;
use rerun::external::re_chunk;
use rerun::external::re_log_types::{EntityPath, TimeInt, TimelineName};
use rerun::external::re_viewer_context::ViewerContext;

/// The source file this recording came from, as logged by the feed.
///
/// The annotator has no idea what it is looking at otherwise -- the feed knows
/// the bag path, so it writes it into the recording for us to read back.
pub fn source_path(ctx: &ViewerContext<'_>) -> Option<String> {

    let entity = EntityPath::from(SOURCE_ENTITY);
    ctx.recording()
        .latest_at_component::<rerun::components::Text>(
            &entity,
            &ctx.current_query(),
            rerun::TextDocument::descriptor_text().component,
        )
        .map(|(_, text)| text.to_string())
}

/// Entity the feed records the source file at.
pub const SOURCE_ENTITY: &str = "meta/source";

/// Sidecar name for a bag, alongside it so the two travel together.
pub fn sidecar_for(bag: &str) -> PathBuf {
    let path = Path::new(bag);
    let stem = path.file_stem().map_or_else(
        || "annotations".to_owned(),
        |s| s.to_string_lossy().into_owned(),
    );
    path.with_file_name(format!("{stem}.labels.jsonl"))
}

/// One box at one instant.
struct Row {
    center: [f32; 3],
    half_size: [f32; 3],
    quaternion: [f32; 4],
    class_id: Option<u16>,
    /// False for a partial write that carried no box geometry.
    has_geometry: bool,
}

/// Collect every stored pose for one annotation entity, keyed by time.
///
/// `None` as a key is static data, which applies at all times.
fn rows_for(
    ctx: &ViewerContext<'_>,
    entity: &EntityPath,
    timeline: TimelineName,
) -> BTreeMap<Option<i64>, Row> {

    let centers = rerun::Boxes3D::descriptor_centers().component;
    let half_sizes = rerun::Boxes3D::descriptor_half_sizes().component;
    let quaternions = rerun::Boxes3D::descriptor_quaternions().component;
    let class_ids = rerun::Boxes3D::descriptor_class_ids().component;

    // Events are gathered and then folded in (time, row id) order, so the last
    // write at each timestamp wins. A clear counts as an event that removes the
    // timestamp: reserved box slots are logged and immediately cleared, and
    // ignoring tombstones exported all 64 of them as real annotations.
    let mut events: Vec<(Option<i64>, re_chunk::RowId, Option<Row>)> = Vec::new();
    let engine = ctx.recording().storage_engine();
    let clear_marker = rerun::archetypes::Clear::descriptor_is_recursive().component;

    for chunk in engine.store().iter_physical_chunks() {
        if chunk.entity_path() != entity {
            continue;
        }

        let times: Vec<Option<i64>> = if chunk.is_static() {
            vec![None; chunk.num_rows()]
        } else {
            let Some(column) = chunk.timelines().get(&timeline) else {
                continue;
            };
            column.times().map(|t: TimeInt| Some(t.as_i64())).collect()
        };

        let row_ids: Vec<re_chunk::RowId> = chunk.row_ids().collect();
        let is_clear = chunk.components().contains_key(&clear_marker);

        let mut center_iter = chunk.iter_slices::<[f32; 3]>(centers);
        let mut half_iter = chunk.iter_slices::<[f32; 3]>(half_sizes);
        let mut quat_iter = chunk.iter_slices::<[f32; 4]>(quaternions);
        let mut class_iter = chunk.iter_slices::<u16>(class_ids);

        for (index, time) in times.into_iter().enumerate() {
            let row_id = row_ids.get(index).copied().unwrap_or(re_chunk::RowId::ZERO);
            if is_clear {
                events.push((time, row_id, None));
                continue;
            }

            let center = center_iter.next().and_then(|s| s.first().copied());
            let half = half_iter.next().and_then(|s| s.first().copied());
            let quat = quat_iter.next().and_then(|s| s.first().copied());
            let class = class_iter.next().and_then(|s| s.first().copied());

            events.push((
                time,
                row_id,
                Some(Row {
                    center: center.unwrap_or([0.0; 3]),
                    half_size: half.unwrap_or([0.0; 3]),
                    quaternion: quat.unwrap_or([0.0, 0.0, 0.0, 1.0]),
                    class_id: class,
                    has_geometry: half.is_some(),
                }),
            ));
        }
    }

    events.sort_by_key(|(time, row_id, _)| (*time, *row_id));

    let mut out: BTreeMap<Option<i64>, Row> = BTreeMap::new();
    for (time, _, row) in events {
        match row {
            None => {
                out.remove(&time);
            }
            Some(row) => {
                // A partial write -- a class change, say -- carries no geometry;
                // fold it onto whatever that timestamp already had.
                let merged = match out.get(&time) {
                    Some(existing) if !row.has_geometry => Row {
                        center: existing.center,
                        half_size: existing.half_size,
                        quaternion: existing.quaternion,
                        class_id: row.class_id.or(existing.class_id),
                        has_geometry: true,
                    },
                    _ => row,
                };
                if merged.has_geometry {
                    out.insert(time, merged);
                }
            }
        }
    }

    // Rerun's own rule: static data outranks temporal for the same component.
    // Ticking a box static leaves its earlier per-frame writes in the store, and
    // exporting both produced two SceneUpdate entities sharing one track id --
    // one of them missing the class, which is only ever written temporally.
    if let Some(mut fixed) = out.remove(&None) {
        if fixed.class_id.is_none() {
            fixed.class_id = out.values().rev().find_map(|row| row.class_id);
        }
        out.clear();
        out.insert(None, fixed);
    }

    out
}

/// Write every annotation under `prefix` to `path`.
///
/// Returns the number of records written.
pub fn export(
    ctx: &ViewerContext<'_>,
    prefix: &EntityPath,
    timeline: TimelineName,
    ontology: &[(u16, String)],
    path: &Path,
) -> std::io::Result<usize> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);

    // Header first: without the frame and the ontology these numbers cannot be
    // interpreted, and both have already changed once during development.
    writeln!(
        file,
        r#"{{"schema":"vision3d.annotations/1","frame":"map","timeline":"{}","classes":[{}]}}"#,
        timeline.as_str(),
        ontology
            .iter()
            .map(|(id, name)| format!(r#"{{"id":{id},"name":"{name}"}}"#))
            .collect::<Vec<_>>()
            .join(",")
    )?;

    let entities: Vec<EntityPath> = ctx
        .recording()
        .sorted_entity_paths()
        .filter(|path| path.starts_with(prefix) && *path != prefix)
        .cloned()
        .collect();

    let mut written = 0;
    for entity in entities {
        let track = entity.last().map(|p| p.ui_string()).unwrap_or_default();
        for (time, row) in rows_for(ctx, &entity, timeline) {
            let class = row
                .class_id
                .and_then(|id| ontology.iter().find(|(cid, _)| *cid == id))
                .map(|(_, name)| name.as_str());

            writeln!(
                file,
                r#"{{"track":"{track}","t":{},"static":{},"class_id":{},"class":{},"center":[{:.4},{:.4},{:.4}],"half_size":[{:.4},{:.4},{:.4}],"quat":[{:.6},{:.6},{:.6},{:.6}]}}"#,
                time.map_or("null".to_owned(), |t| t.to_string()),
                time.is_none(),
                row.class_id
                    .map_or("null".to_owned(), |id| id.to_string()),
                class.map_or("null".to_owned(), |c| format!("\"{c}\"")),
                row.center[0],
                row.center[1],
                row.center[2],
                row.half_size[0],
                row.half_size[1],
                row.half_size[2],
                row.quaternion[0],
                row.quaternion[1],
                row.quaternion[2],
                row.quaternion[3],
            )?;
            written += 1;
        }
    }

    file.flush()?;
    re_log::info!("exported {written} annotation record(s) to {}", path.display());
    Ok(written)
}


/// Status of a save that is running on a background thread.
static SAVE_STATUS: Mutex<Option<String>> = Mutex::new(None);

pub fn save_status() -> Option<String> {
    SAVE_STATUS.lock().clone()
}

/// Write the exported labels into the bag itself, off the UI thread.
///
/// This shells out to `save_labels.py` rather than writing MCAP here. Doing it
/// natively would mean an MCAP writer *and* a CDR encoder for SceneUpdate in
/// Rust; the Python path gets both from `mcap_ros2` and is already tested
/// against real recordings. The cost is that the annotator needs the project's
/// Python environment, which is fine for development but is exactly what a
/// standalone build for external annotators would have to remove.
///
/// The rewrite takes several seconds on a 900 MB bag, so it runs on its own
/// thread; blocking here would freeze the viewer mid-save.
pub fn save_into_bag(bag: &str, sidecar: &Path) {
    let bag = bag.to_owned();
    let sidecar = sidecar.to_path_buf();

    *SAVE_STATUS.lock() = Some("writing into bag…".to_owned());

    std::thread::spawn(move || {
        // The script lives beside this crate; run it from the repo root, which
        // is where its relative imports and `uv` project resolve from.
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let script = manifest.join("save_labels.py");
        let repo_root = manifest.parent().and_then(Path::parent);

        let mut command = std::process::Command::new("uv");
        command
            .arg("run")
            .arg("python")
            .arg(&script)
            .arg("--bag")
            .arg(&bag)
            .arg("--labels")
            .arg(&sidecar);
        if let Some(root) = repo_root {
            command.current_dir(root);
        }

        let status = match command.output() {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let summary = stdout
                    .lines()
                    .last()
                    .map_or_else(|| "written into bag".to_owned(), str::to_owned);

                // The sidecar was only ever a hand-off to the writer; the bag is
                // the record now, and leaving it behind invites editing the wrong
                // one. Only removed once the write reported success.
                if let Err(err) = std::fs::remove_file(&sidecar) {
                    re_log::warn!("could not remove {}: {err}", sidecar.display());
                }
                summary
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                re_log::error!("save_labels.py failed: {stderr}");
                format!(
                    "save failed: {}",
                    stderr.lines().last().unwrap_or("see log")
                )
            }
            Err(err) => {
                re_log::error!("could not run save_labels.py: {err}");
                format!("could not run save_labels.py: {err}")
            }
        };

        *SAVE_STATUS.lock() = Some(status);
    });
}
