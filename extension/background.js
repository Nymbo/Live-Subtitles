// Live Subtitles: relays the page's video state to the app on 127.0.0.1, with the stream
// and subtitle URLs this tab's player requested (the app needs them to make subtitles
// ahead of playback). Nothing leaves this machine.

importScripts("classify.js"); // MEDIA, SUBS, subtitleOf

const DEFAULT_PORT = 47219;
const seen = new Map(); // tabId -> {media: [{url, type, frameId}], subtitles: [{url, name, frameId}]}
const pages = new Map(); // tabId -> Map(frameId -> {url, names, frames}): what each frame calls the programme

function remember(tabId, list, item) {
  if (tabId < 0) return;
  const entry = seen.get(tabId) || { media: [], subtitles: [] };
  const arr = entry[list];
  if (!arr.some((x) => x.url === item.url)) {
    if (list === "subtitles") {
      // Players often fetch every language at once; keep them in request order (the
      // site's own order, English usually first) and enough of them to reach English.
      arr.push(item);
      arr.length = Math.min(arr.length, 200);
    } else {
      arr.unshift(item);
      arr.length = Math.min(arr.length, 12);
    }
  }
  seen.set(tabId, entry);
}

function classify(tabId, frameId, url) {
  const sub = subtitleOf(url);
  if (sub) {
    remember(tabId, "subtitles", { ...sub, frameId });
  } else if (MEDIA.test(url) && !/\.ts(\?|$)/i.test(url)) {
    const type = /\.m3u8/i.test(url) ? "hls" : /\.mpd/i.test(url) ? "dash" : "file";
    // A variant playlist arrives after its master; keep the first (master) per tab.
    remember(tabId, "media", { url, type, frameId });
  }
}

chrome.webRequest.onBeforeRequest.addListener(
  (d) => {
    if (d.tabId >= 0) {
      classify(d.tabId, d.frameId, d.url);
    } else if (d.initiator && (SUBS.test(d.url) || subtitleOf(d.url))) {
      // Fetched by the site's own service worker: no tab, so credit the tabs on that site.
      chrome.tabs.query({}, (tabs) => {
        for (const t of tabs) {
          if (t.url && t.url.startsWith(d.initiator)) classify(t.id, 0, d.url);
        }
      });
    }
  },
  { urls: ["<all_urls>"], types: ["xmlhttprequest", "media", "other"] }
);

// A reloaded or updated extension isn't in the tabs already open until they reload; put the
// content script in them now, so a video that is already playing is followed straight away.
chrome.runtime.onInstalled.addListener(() => {
  chrome.tabs.query({ url: ["http://*/*", "https://*/*"] }, (tabs) => {
    for (const t of tabs) {
      chrome.scripting
        .executeScript({ target: { tabId: t.id, allFrames: true }, files: ["classify.js", "content.js"] })
        .catch(() => {}); // pages the extension may not touch (the web store, error pages)
    }
  });
});

chrome.tabs.onRemoved.addListener((tabId) => {
  seen.delete(tabId);
  pages.delete(tabId);
});
chrome.tabs.onUpdated.addListener((tabId, info) => {
  if (info.status === "loading" && info.url) {
    seen.delete(tabId);
    pages.delete(tabId);
  }
});

function notePage(tabId, frameId, msg) {
  const frames = pages.get(tabId) || new Map();
  frames.set(frameId, { url: msg.frame || "", names: msg.names || [], frames: msg.frames || [] });
  pages.set(tabId, frames);
}

// Every frame's names, the top page's first (its heading names the film; the player's
// frame is often titled after the player), and every frame address.
function pageOf(tabId, videoFrameId) {
  const frames = [...(pages.get(tabId) || new Map()).entries()]
    .sort(([a], [b]) => (a === 0 ? -1 : b === 0 ? 1 : a === videoFrameId ? -1 : b === videoFrameId ? 1 : 0));
  const names = [...new Set(frames.flatMap(([, f]) => f.names))].slice(0, 16);
  const urls = [...new Set(frames.flatMap(([, f]) => [f.url, ...f.frames]).filter(Boolean))].slice(0, 24);
  return { names, urls };
}

async function port() {
  try {
    const { port } = await chrome.storage.local.get("port");
    return port || DEFAULT_PORT;
  } catch (e) {
    return DEFAULT_PORT;
  }
}

chrome.runtime.onMessage.addListener((msg, sender) => {
  if (!msg || !sender.tab) return;
  const tabId = sender.tab.id;
  if (msg.type === "files") {
    for (const url of msg.loaded || []) classify(tabId, sender.frameId, url);
    if (msg.names || msg.frames) notePage(tabId, sender.frameId, msg);
    return;
  }
  if (msg.type !== "video") return;
  notePage(tabId, sender.frameId, msg);
  // Files the page loaded before this worker saw its requests (extension reloaded, worker restarted).
  for (const url of msg.loaded || []) classify(tabId, sender.frameId, url);
  const entry = seen.get(tabId) || { media: [], subtitles: [] };
  // Media requested by the frame that holds the video first; its URL is the Referer the CDN expects.
  const sameFrame = (x) => x.frameId === sender.frameId;
  const ordered = [...entry.media].reverse(); // oldest first: the master playlist before its variants
  const media = [...ordered.filter(sameFrame), ...ordered.filter((x) => !sameFrame(x))]
    .map((x) => ({ url: x.url, type: x.type }));
  if (msg.src && !media.some((m) => m.url === msg.src)) media.unshift({ url: msg.src, type: "file" });
  const body = {
    title: sender.tab.title || "",
    tab: sender.tab.url || "",
    tab_id: tabId,
    // Several tabs can hold a playing video; the app follows the one making sound.
    audible: !!sender.tab.audible,
    muted: !!(sender.tab.mutedInfo && sender.tab.mutedInfo.muted),
    page: msg.frame,
    position: msg.position,
    duration: msg.duration,
    paused: msg.paused,
    rate: msg.rate,
    reason: msg.reason,
    started_ago: msg.started_ago,
    visible: msg.visible !== false && sender.tab.active,
    episode: msg.episode || null,
    ...pageOf(tabId, sender.frameId),
    media,
    subtitles: entry.subtitles.map((x) => ({ url: x.url, name: x.name || "" })),
  };
  port().then((p) =>
    fetch(`http://127.0.0.1:${p}/player`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Live-Subtitles": "1" },
      body: JSON.stringify(body),
    }).catch(() => {}) // the app isn't running; try again on the next report
  );
});
