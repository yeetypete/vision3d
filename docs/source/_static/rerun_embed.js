// Initialize every <div class="rerun-embed" data-rrd="..."> on the page with
// the @rerun-io/web-viewer SDK.
//
// The SDK version is read from the first embed's data-rerun-version attribute
// (set by the rerun-embed directive from the installed rerun.__version__),
// keeping the embedded WebViewer pinned to the SDK that wrote the .rrd.

async function initRerunEmbeds() {
  const containers = document.querySelectorAll(".rerun-embed[data-rrd]");
  if (containers.length === 0) return;

  // eframe's text agent (a hidden <input> at top:0;left:0 appended to
  // <body> on init, see eframe/src/web/text_agent.rs) is created with
  // the `autofocus` HTML attribute. The browser's autofocus algorithm
  // focuses the input as soon as it's connected and scrolls the page
  // to bring it into view (the input sits at (0,0)), so every page
  // load and refresh of an embed page snaps to the top - defeating
  // the browser's automatic scroll restoration. Browser-native
  // autofocus bypasses the focus() prototype patch below.
  //
  // Shadow the `autofocus` IDL setter on HTMLInputElement: eframe sets
  // autofocus via `input.autofocus = true` (wasm-bindgen
  // `__wbg_set_autofocus_*`), and shadowing the property setter means
  // the assignment is a no-op so the browser never marks the input as
  // an autofocus candidate. The page no longer scrolls on init.
  Object.defineProperty(HTMLInputElement.prototype, "autofocus", {
    get() {
      return false;
    },
    set() {},
    configurable: true,
  });

  const version = containers[0].dataset.rerunVersion;
  const { WebViewer } = await import(
    `https://cdn.jsdelivr.net/npm/@rerun-io/web-viewer@${version}/+esm`
  );

  for (const el of containers) {
    const rrdUrl = new URL(el.dataset.rrd, document.baseURI).href;
    const viewer = new WebViewer();
    await viewer.start(rrdUrl, el, {
      width: "100%",
      height: "100%",
      allow_fullscreen: true,
    });
  }
}

initRerunEmbeds();
