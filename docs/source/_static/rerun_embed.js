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

  const version = containers[0].dataset.rerunVersion;
  const { WebViewer } = await import(
    `https://cdn.jsdelivr.net/npm/@rerun-io/web-viewer@${version}/+esm`
  );

  const theme = rerunThemeFromDocs();
  for (const el of containers) {
    const rrdUrl = new URL(el.dataset.rrd, document.baseURI).href;
    await new WebViewer().start(rrdUrl, el, {
      width: "100%",
      height: "100%",
      allow_fullscreen: true,
      theme,
    });
  }
}

initRerunEmbeds();
