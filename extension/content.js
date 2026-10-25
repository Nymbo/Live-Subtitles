// Live Subtitles: report the page's main video to the extension's background worker.
// Runs in every frame, because streaming sites play the video inside an iframe from
// another domain. Only a real programme counts (two minutes or longer), not previews or ads.

(() => {
  // Injected again after an extension reload: stop the copy that was here before.
  if (window.__liveSubtitlesStop) window.__liveSubtitlesStop();
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

  // Sites like Peacock and Netflix keep one page title for a whole series and show the
  // episode only in the player ("S2 E3", "S2:E3", "Season 2, Episode 3"). Read it from the
  // media session, then from the nearest player container that names one.
  // No word boundary in front: textContent runs elements together ("The OfficeS2 E3").
  const EPISODE = /S(\d{1,2})\s*[:.,·-]?\s*E(?:p(?:isode)?)?\.?\s*(\d{1,3})(?!\d)/;
  const LONG_EPISODE = /Season\s*(\d{1,2})\s*[,:·-]?\s*Episode\s*(\d{1,3})(?!\d)/i;
  let episodeCache = { at: 0, value: null };

  // Subtitle files and streams the page already loaded, re-sent every 10 s so a restarted
  // extension worker learns them again (players fetch subtitles once, at page load).
  try {
    performance.setResourceTimingBufferSize(3000);
  } catch (e) {}
  let loadedAt = 0;
  function loadedFiles() {
    const now = Date.now();
    if (now - loadedAt < 10000) return [];
    loadedAt = now;
    return performance.getEntriesByType("resource").map((e) => e.name)
      .filter((u) => subtitleOf(u) || MEDIA.test(u)).slice(0, 400);
  }

  function findEpisode(text) {
    const m = (text || "").match(LONG_EPISODE) || (text || "").match(EPISODE);
    return m ? { season: Number(m[1]), episode: Number(m[2]) } : null;
  }

  function episodeOf(v) {
    const now = Date.now();
    if (now - episodeCache.at < 5000) return episodeCache.value;
    let found = null;
    try {
      const md = navigator.mediaSession && navigator.mediaSession.metadata;
      if (md) found = findEpisode([md.title, md.artist, md.album].join(" "));
    } catch (e) {}
    for (let el = v.parentElement, depth = 0; !found && el && depth < 12; el = el.parentElement, depth++) {
      found = findEpisode(el.textContent);
    }
    episodeCache = { at: now, value: found };
    return found;
  }

  function report(reason) {
    const v = current;
    if (!v) return;
    const src = v.currentSrc || v.src || "";
    try {
      chrome.runtime.sendMessage({
        type: "video",
        reason,
        episode: episodeOf(v),
        loaded: loadedFiles(),
        position: v.currentTime,
        duration: Number.isFinite(v.duration) ? v.duration : 0,
        paused: v.paused || v.ended,
        rate: v.playbackRate,
        // A video in a background tab or a minimised window isn't the one being watched.
        visible: document.visibilityState === "visible",
        // Seconds since playback started, so the app can tell which tab was started last.
        // A video already playing when this script arrived counts as started long ago.
        started_ago: startedAt === null ? 86400 : (Date.now() - startedAt) / 1000,
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

  let startedAt = null; // when the video last started playing (null: before this script ran)
  function onEvent(e) {
    if (e.type === "play") startedAt = Date.now();
    report(e.type);
  }

  window.__liveSubtitlesStop = () => {
    clearInterval(timer);
    attach(null);
  };
  timer = setInterval(() => {
    attach(mainVideo());
    if (current && !current.paused) report("tick");
    else if (!current) {
      // No video in this frame, but it may be the one that fetched the subtitles.
      const loaded = loadedFiles();
      if (loaded.length) {
        try {
          chrome.runtime.sendMessage({ type: "files", loaded });
        } catch (e) {
          clearInterval(timer);
        }
      }
    }
  }, REPORT_MS);
})();
