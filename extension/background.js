// Live Subtitles: relays the page's video state to the app on 127.0.0.1, with the stream
// and subtitle URLs this tab's player requested (the app needs them to make subtitles
// ahead of playback). Nothing leaves this machine.

const DEFAULT_PORT = 47219;
const MEDIA = /\.(m3u8|mpd|mp4|m4v|webm|mkv)(\?|$)/i;
const SUBS = /\.(vtt|srt|ass|ssa)(\?|$)/i;
const seen = new Map(); // tabId -> {media: [{url, type, frameId}], subtitles: [{url, frameId}]}

function remember(tabId, list, item) {
  if (tabId < 0) return;
  const entry = seen.get(tabId) || { media: [], subtitles: [] };
  const arr = entry[list];
  if (!arr.some((x) => x.url === item.url)) {
    arr.unshift(item);
    arr.length = Math.min(arr.length, 12);
  }
  seen.set(tabId, entry);
}

chrome.webRequest.onBeforeRequest.addListener(
  (d) => {
    const url = d.url;
    if (SUBS.test(url)) remember(d.tabId, "subtitles", { url, frameId: d.frameId });
    else if (MEDIA.test(url) && !/\.ts(\?|$)/i.test(url)) {
      const type = /\.m3u8/i.test(url) ? "hls" : /\.mpd/i.test(url) ? "dash" : "file";
      // A variant playlist arrives after its master; keep the first (master) per tab.
      remember(d.tabId, "media", { url, type, frameId: d.frameId });
    }
  },
  { urls: ["<all_urls>"], types: ["xmlhttprequest", "media", "other"] }
);

chrome.tabs.onRemoved.addListener((tabId) => seen.delete(tabId));
chrome.tabs.onUpdated.addListener((tabId, info) => {
  if (info.status === "loading" && info.url) seen.delete(tabId);
});

async function port() {
  try {
    const { port } = await chrome.storage.local.get("port");
    return port || DEFAULT_PORT;
  } catch (e) {
    return DEFAULT_PORT;
  }
}

chrome.runtime.onMessage.addListener((msg, sender) => {
  if (!msg || msg.type !== "video" || !sender.tab) return;
  const tabId = sender.tab.id;
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
    page: msg.frame,
    position: msg.position,
    duration: msg.duration,
    paused: msg.paused,
    rate: msg.rate,
    reason: msg.reason,
    media,
    subtitles: entry.subtitles.map((x) => ({ url: x.url })),
  };
  port().then((p) =>
    fetch(`http://127.0.0.1:${p}/player`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Live-Subtitles": "1" },
      body: JSON.stringify(body),
    }).catch(() => {}) // the app isn't running; try again on the next report
  );
});
