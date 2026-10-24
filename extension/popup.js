const DEFAULT_PORT = 47219;
const STATES = {
  watching: "Waiting for a movie or episode to play.",
  identifying: "Finding out what's playing…",
  searching: "Finding subtitles…",
  aligning: "Syncing subtitles…",
  synced: "Synced.",
  generating: "Making subtitles from the audio.",
  unknown: "Not sure what's playing. Right-click the subtitles → What is this?",
};

async function refresh() {
  const el = document.getElementById("state");
  const { port = DEFAULT_PORT } = await chrome.storage.local.get("port");
  document.getElementById("port").value = port;
  try {
    const r = await fetch(`http://127.0.0.1:${port}/status`, { headers: { "X-Live-Subtitles": "1" } });
    const s = await r.json();
    if (!s.running) {
      el.textContent = "Media subtitles are off in the app.";
      return;
    }
    el.textContent = [s.title, STATES[s.state] || s.state, s.detail].filter(Boolean).join(" · ");
  } catch (e) {
    el.innerHTML = '<span class="off">The Live Subtitles app isn\'t running.</span>';
  }
}

document.getElementById("port").addEventListener("change", async (e) => {
  const port = parseInt(e.target.value, 10);
  if (port >= 1024 && port <= 65535) await chrome.storage.local.set({ port });
  refresh();
});

refresh();
