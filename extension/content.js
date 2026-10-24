// Live Subtitles: report the page's main video to the extension's background worker.
// Runs in every frame, because streaming sites play the video inside an iframe from
// another domain. Only a real programme counts (two minutes or longer), not previews or ads.

(() => {
  const MIN_DURATION_S = 120;
  const REPORT_MS = 500;
  let current = null;
  let timer = null;

  function mainVideo() {
    let best = null;
    let bestArea = 0;
    for (const v of document.querySelectorAll("video")) {
      const d = v.duration;
      if (!(d >= MIN_DURATION_S) && d !== Infinity) continue;
      const r = v.getBoundingClientRect();
      const area = r.width * r.height;
      if (area > bestArea) {
        best = v;
        bestArea = area;
      }
    }
    return best;
  }

  function report(reason) {
    const v = current;
    if (!v) return;
    const src = v.currentSrc || v.src || "";
    try {
      chrome.runtime.sendMessage({
        type: "video",
        reason,
        position: v.currentTime,
        duration: Number.isFinite(v.duration) ? v.duration : 0,
        paused: v.paused || v.ended,
        rate: v.playbackRate,
        src: src.startsWith("blob:") ? "" : src,
        frame: location.href,
      });
    } catch (e) {
      // The extension was reloaded; this old content script is orphaned.
      clearInterval(timer);
    }
  }

  const EVENTS = ["play", "pause", "seeked", "ratechange", "loadedmetadata", "ended"];

  function attach(v) {
    if (v === current) return;
    if (current) EVENTS.forEach((e) => current.removeEventListener(e, onEvent));
    current = v;
    if (v) EVENTS.forEach((e) => v.addEventListener(e, onEvent));
  }

  function onEvent(e) {
    report(e.type);
  }

  timer = setInterval(() => {
    attach(mainVideo());
    if (current && !current.paused) report("tick");
  }, REPORT_MS);
})();
