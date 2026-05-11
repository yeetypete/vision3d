// Initialize every <div class="rerun-embed" data-rrd="..."> on the page with
// the @rerun-io/web-viewer SDK.
//
// The SDK version is read from the first embed's data-rerun-version attribute
// (set by the rerun-embed directive from the installed rerun.__version__),
// keeping the embedded WebViewer pinned to the SDK that wrote the .rrd.

function rerunThemeFromDocs() {
  const mode = document.documentElement.dataset.mode;
  if (mode === "light" || mode === "dark") return mode;
  return "system";
}

async function initRerunEmbeds() {
  const containers = document.querySelectorAll(".rerun-embed[data-rrd]");
  if (containers.length === 0) return;

  // The rerun web viewer is built on eframe, whose text agent is a
  // 1x1 hidden <input> with `autofocus` and `position: absolute`
  // (eframe/src/web/text_agent.rs). On page load the autofocus
  // algorithm scrolls the page to where the input sits. Once focused,
  // eframe also moves it via style.top to track the egui caret, and
  // the browser auto-scrolls the page to keep the focused input
  // visible. Pin it to the viewport to defuse both.
  // Related: https://github.com/emilk/egui/issues/7887
  new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (
          node.tagName === "INPUT" &&
          node.style.width === "1px" &&
          node.style.height === "1px"
        ) {
          node.style.setProperty("position", "fixed", "important");
        }
      }
    }
  }).observe(document.body, { childList: true });

  const version = containers[0].dataset.rerunVersion;
  const { WebViewer } = await import(
    `https://cdn.jsdelivr.net/npm/@rerun-io/web-viewer@${version}/+esm`
  );

  const theme = rerunThemeFromDocs();
  for (const el of containers) {
    const rrdUrl = new URL(el.dataset.rrd, document.baseURI).href;
    const viewer = new WebViewer();
    // start() creates and appends the canvas synchronously before its
    // first await, so viewer.canvas is available immediately. eframe
    // re-focuses the canvas on every repaint when not in IME mode
    // (eframe/src/web/app_runner.rs:395-405), so each keystroke pulls
    // the page to wherever the canvas sits unless we default focus() to
    // preventScroll. Patch between start() and its await so the WASM
    // never sees the unpatched method.
    const startPromise = viewer.start(rrdUrl, el, {
      width: "100%",
      height: "100%",
      allow_fullscreen: true,
      theme,
    });
    const canvas = viewer.canvas;
    if (canvas) {
      const orig = canvas.focus.bind(canvas);
      canvas.focus = (opts) => orig({ ...opts, preventScroll: true });
    }
    await startPromise;
  }
}

initRerunEmbeds();
