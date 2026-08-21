//! Choosing a recording and loading it, by driving the Python feed.
//!
//! Reading MCAP happens in Python, so this spawns the feed rather than parsing
//! anything here. Same trade as the export path: it reuses code already tested
//! against real recordings, at the cost of needing the project's Python
//! environment. A standalone build for external annotators would have to move
//! both into this binary.

use std::path::{Path, PathBuf};

use rerun::external::parking_lot::Mutex;
use rerun::external::re_log;

static STATUS: Mutex<Option<String>> = Mutex::new(None);
static OPTIONS: Mutex<LoadOptions> = Mutex::new(LoadOptions::DEFAULT);

/// How much of a recording to read, and how finely.
#[derive(Clone, Copy)]
pub struct LoadOptions {
    /// Keyframe rate. The lidars publish at about 10 Hz, so 10 uses every sweep.
    pub hz: f32,
    /// Past captures kept for the sweeps slider.
    pub sweeps: u32,
    /// Read to the end of the bag rather than `seconds` from `start_at`.
    pub whole: bool,
    /// Length to read when not reading the whole bag.
    pub seconds: f32,
    /// Offset into the bag to start at, in seconds. Free to change: the reader
    /// seeks using the bag's chunk index rather than reading up to it.
    pub start_at: f32,
}

impl LoadOptions {
    const DEFAULT: Self = Self {
        hz: 10.0,
        sweeps: 5,
        whole: true,
        seconds: 20.0,
        start_at: 0.0,
    };
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

pub fn options() -> LoadOptions {
    *OPTIONS.lock()
}

pub fn set_options(options: LoadOptions) {
    *OPTIONS.lock() = options;
}

/// What the loader is doing, for display next to the button.
pub fn status() -> Option<String> {
    STATUS.lock().clone()
}

/// Ask for a recording and stream it in.
///
/// The dialog and the feed both run off the UI thread: a modal dialog would
/// block rendering, and a full bag takes a while to read.
pub fn pick_and_load(default_dir: Option<&Path>) {
    let default_dir = default_dir.map(Path::to_path_buf);

    std::thread::spawn(move || {
        let mut dialog = rfd::FileDialog::new().add_filter("MCAP recording", &["mcap"]);
        if let Some(dir) = default_dir.filter(|d| d.is_dir()) {
            dialog = dialog.set_directory(dir);
        }

        let Some(bag) = dialog.pick_file() else {
            return; // cancelled
        };
        run_feed(&bag);
    });
}

/// Load a specific recording, skipping the dialog.
pub fn load_path(bag: PathBuf) {
    std::thread::spawn(move || run_feed(&bag));
}

/// Stream one recording in, reporting progress through [`status`].
fn run_feed(bag: &Path) {
    *STATUS.lock() = Some(format!(
        "loading {}…",
        bag.file_name().unwrap_or_default().to_string_lossy()
    ));

    let options = options();
    {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let script = manifest.join("feed_mcap.py");
        let repo_root = manifest.parent().and_then(Path::parent);

        let mut command = std::process::Command::new("uv");
        command
            .arg("run")
            .arg("python")
            .arg(&script)
            .arg("--bag")
            .arg(bag)
            .arg("--hz")
            .arg(options.hz.to_string())
            .arg("--num-sweeps")
            .arg(options.sweeps.to_string());
        if options.whole {
            command.arg("--all");
        } else {
            command.arg("--seconds").arg(options.seconds.to_string());
        }
        if options.start_at > 0.0 {
            command.arg("--start-at").arg(options.start_at.to_string());
        }
        if let Some(root) = repo_root {
            command.current_dir(root);
        }

        let status = match command.output() {
            Ok(output) if output.status.success() => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout
                    .lines()
                    .filter(|line| line.contains("keyframes"))
                    .next_back()
                    .map_or_else(|| "loaded".to_owned(), str::to_owned)
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                re_log::error!("feed_mcap.py failed: {stderr}");
                format!("load failed: {}", stderr.lines().last().unwrap_or("see log"))
            }
            Err(err) => {
                re_log::error!("could not run feed_mcap.py: {err}");
                format!("could not run feed_mcap.py: {err}")
            }
        };

        *STATUS.lock() = Some(status);
    }
}

/// Directory to open the dialog in: wherever the current recording came from.
pub fn default_dir(source: Option<&str>) -> Option<PathBuf> {
    source
        .map(Path::new)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
}
