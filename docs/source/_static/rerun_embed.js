// Initialize every <div class="rerun-embed" data-rrd="..."> on the page with
// the @rerun-io/web-viewer SDK.
//
// The SDK version is read from the first embed's data-rerun-version attribute
// (set by the rerun-embed directive from the installed rerun.__version__),
// keeping the embedded WebViewer pinned to the SDK that wrote the .rrd.

async function initRerunEmbeds() {
  const containers = document.querySelectorAll(".rerun-embed[data-rrd]");
  if (containers.length === 0) return;

  const version = containers[0].dataset.rerunVersion;
  const { WebViewer } = await import(
    `https://cdn.jsdelivr.net/npm/@rerun-io/web-viewer@${version}/+esm`
  );

  for (const el of containers) {
    const rrdUrl = new URL(el.dataset.rrd, document.baseURI).href;
    const viewer = new WebViewer();
    // start() creates and appends the canvas synchronously before its
    // first await, so viewer.canvas is set as soon as the call returns
    // a promise. Patch focus() on the canvas to default preventScroll:
    // true before the WASM (eframe) calls it on mount, otherwise Chrome
    // scrolls the page to bring the focused canvas into view, jumping
    // to the embed. Firefox does not exhibit this. The WebViewer API
    // exposes no option to control this upstream.
    const startPromise = viewer.start(rrdUrl, el, {
      width: "100%",
      height: "100%",
      allow_fullscreen: true,
    });
    const canvas = viewer.canvas;
    if (canvas) {
      const origFocus = canvas.focus.bind(canvas);
      canvas.focus = (opts) => origFocus({ ...opts, preventScroll: true });
    }
    await startPromise;
  }
}

initRerunEmbeds();
