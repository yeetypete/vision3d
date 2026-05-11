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
  // hidden <input> at top:0;left:0 in <body> created with `autofocus`
  // (eframe/src/web/text_agent.rs:25). The browser's autofocus
  // algorithm scrolls the page to (0,0) on insertion, defeating scroll
  // restoration on every embed page load. We shadow the IDL setter so
  // eframe's `input.autofocus = true` is a no-op.
  // Related: https://github.com/emilk/egui/issues/7887
  Object.defineProperty(HTMLInputElement.prototype, "autofocus", {
    get() {
      return false;
    },
    set() {},
    configurable: true,
  });

  // eframe re-focuses its hidden text-agent <input> on every keystroke
  // as a workaround for an Android Gboard issue
  // (eframe/src/web/text_agent.rs:67-68). Without `preventScroll: true`,
  // Chrome scrolls the page to bring the input (pinned at top:0;left:0)
  // into view, so every keystroke jerks the page to the top. Fix: Match
  // the text agent by its 1x1 inline width/height and patch its focus() to
  // default preventScroll, leaving unrelated inputs (Sphinx search, etc.)
  // untouched.
  // Related: https://github.com/emilk/egui/issues/7887
  new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (
          node.tagName === "INPUT" &&
          node.style.width === "1px" &&
          node.style.height === "1px"
        ) {
          const orig = node.focus.bind(node);
          node.focus = (opts) => orig({ ...opts, preventScroll: true });
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
