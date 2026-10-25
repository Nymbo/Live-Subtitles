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

  // What the page calls the programme. Sites put it in different places: the tab title,
  // structured data, share-card metadata, the media session, a heading, or only the
  // player's address with the tab titled after the site. Report them all, best first; the
  // app skips the ones that only name the site. Frame addresses carry embed players' ids.
  let namesCache = { at: 0, value: { names: [], frames: [] } };
  let namesSentAt = 0;
  function fromJsonLd(node, out) {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) return node.forEach((n) => fromJsonLd(n, out));
    if (node["@graph"]) fromJsonLd(node["@graph"], out);
    const types = [].concat(node["@type"] || []);
    const year = String(node.datePublished || node.dateCreated || "").slice(0, 4);
    if (types.includes("TVEpisode") && node.episodeNumber) {
      const series = (node.partOfSeries || node.partOfTVSeries || {}).name || "";
      const season = (node.partOfSeason || {}).seasonNumber || 1;
      if (series) out.push(`${series} S${season}E${node.episodeNumber}`);
    } else if (types.some((t) => t === "Movie" || t === "TVSeries" || t === "VideoObject") && node.name) {
      out.push(/^\d{4}$/.test(year) && types.includes("Movie") ? `${node.name} (${year})` : node.name);
    }
  }
  const big = (r) => r.width >= 200 && r.height >= 120;
  // The page itself, or a frame big enough to hold a player: not an ad or tracker frame.
  const PAGE_FRAME = window === window.top || big({ width: window.innerWidth, height: window.innerHeight });
  function pageNames() {
    const now = Date.now();
    if (now - namesCache.at < 3000) return namesCache.value;
    const names = [];
    for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
      try {
        fromJsonLd(JSON.parse(s.textContent), names);
      } catch (e) {}
    }
    for (const sel of ['meta[property="og:title"]', 'meta[name="twitter:title"]', 'meta[name="title"]']) {
      const c = document.querySelector(sel);
      if (c && c.content) names.push(c.content);
    }
    try {
      const md = navigator.mediaSession && navigator.mediaSession.metadata;
      if (md && md.title) names.push(md.title);
    } catch (e) {}
    for (const h of document.querySelectorAll("h1, h2")) {
      const text = (h.textContent || "").replace(/\s+/g, " ").trim(); // not innerText: CSS capitals
      if (text.length >= 2 && text.length <= 120 && h.getClientRects().length) {
        names.push(text);
        break;
      }
    }
    if (document.title) names.push(document.title);
    // Frames big enough to be a player; ads and tracking pixels are tiny or hidden.
    const frames = [...document.querySelectorAll("iframe")]
      .filter((f) => big(f.getBoundingClientRect()))
      .map((f) => f.src).filter((u) => /^https?:/.test(u));
    namesCache = { at: now, value: { names: [...new Set(names)].slice(0, 10), frames: frames.slice(0, 10) } };
    return namesCache.value;
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
        ...pageNames(),
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
      // No video in this frame, but it may be the one that fetched the subtitles, or the
      // page around the player that names the film.
      const loaded = loadedFiles();
      const page = PAGE_FRAME ? pageNames() : { names: [], frames: [] };
      const fresh = namesCache.at !== namesSentAt; // names are re-read every 3 s; send each reading once
      namesSentAt = namesCache.at;
      if (loaded.length || (fresh && (page.names.length || page.frames.length))) {
        try {
          chrome.runtime.sendMessage({ type: "files", loaded, frame: PAGE_FRAME ? location.href : "", ...page });
        } catch (e) {
          clearInterval(timer);
        }
      }
    }
  }, REPORT_MS);
})();
