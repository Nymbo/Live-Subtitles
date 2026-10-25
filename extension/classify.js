// Live Subtitles: which requested URLs are subtitle files or video streams. Shared by the
// background worker (requests as they happen) and the content script (files the page had
// already loaded before the extension started, from the page's resource timing list).
// Top-level "var" and functions only: the script can be injected twice into one frame.

var MEDIA = /\.(m3u8|mpd|mp4|m4v|webm|mkv)(\?|$)/i;
var SUBS = /\.(vtt|srt|ass|ssa)(\?|$)/i;

function fileName(url) {
  try {
    return decodeURIComponent(url.split("?")[0].split("/").pop());
  } catch (e) {
    return url.split("?")[0].split("/").pop();
  }
}

// Some sites fetch subtitles through a proxy that base64-encodes the real file's address
// into the path (".../sub/aHR0cHM6Ly9...c3J0"). Returns the wrapped file name, or "".
function wrappedSubtitle(url) {
  let parts;
  try {
    const u = new URL(url);
    parts = [...u.pathname.split("/"), ...u.searchParams.values()];
  } catch (e) {
    return "";
  }
  for (const p of parts) {
    if (p.length < 24 || !/^[A-Za-z0-9+/_=-]+$/.test(p)) continue;
    try {
      const b64 = p.replace(/-/g, "+").replace(/_/g, "/");
      const inner = atob(b64 + "=".repeat((4 - (b64.length % 4)) % 4));
      if (/^https?:\/\//.test(inner) && SUBS.test(inner)) return fileName(inner);
    } catch (e) {}
  }
  return "";
}

// {url, name} for a subtitle file, or null.
function subtitleOf(url) {
  if (SUBS.test(url)) return { url, name: fileName(url) };
  const wrapped = wrappedSubtitle(url);
  return wrapped ? { url, name: wrapped } : null;
}
