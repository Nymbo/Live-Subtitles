# Store listing (Chrome Web Store and Edge Add-ons)

Copy these into the dashboard. Build the upload with `python tools/extension.py pack`.

## Store listing

**Name:** Live Subtitles

**Summary (132 characters max):**
Companion to the Live Subtitles desktop app: tells it where the video on the page is, so movie and TV subtitles line up exactly.

**Category:** Accessibility

**Language:** English

**Description:**

Live Subtitles puts real subtitles on whatever you watch in your browser: whole lines, on
time, ahead of the speech, instead of live captions that trail behind it.

This extension is the browser half of the free Live Subtitles app for Windows
(https://github.com/Nymbo/Live-Subtitles). The app finds subtitles for the movie or episode
that is playing and syncs them to what it hears. With the extension, timing is exact from
the first line and survives seeks, pauses and skipped scenes, because the app knows the
video's real position.

What the extension does:
• Reports the playing video's position, pause state and length to the app on your computer.
• Reads the season and episode off the player on sites whose page title doesn't say which
  episode is on.
• Passes along the page player's own subtitle files, so the app can use subtitles made for
  that exact stream.
• Follows the tab you are watching when videos play in several tabs.

Everything stays on your computer: the extension only talks to the app at 127.0.0.1. No
account, no analytics, no ads.

Requires the Live Subtitles app for Windows. Without the app running, the extension does
nothing.

## Privacy practices

**Single purpose:**
Report the state of the video playing on the page (position, pause, episode, and the
player's own subtitle files) to the Live Subtitles desktop app on the same computer, so the
app can show subtitles in sync with the video.

**Permission justifications:**

- `webRequest`: Sees the addresses of subtitle files and video streams the page's player
  requests, so the app can use the player's own subtitles (made for that exact stream) and
  generate subtitles ahead of playback. Requests are only observed, never blocked or changed.
- `scripting`: Adds the content script to tabs that were already open when the extension was
  installed or updated, so a video that is already playing is followed without reloading.
- `storage`: Remembers one setting, the port the desktop app listens on.
- Host permission `<all_urls>`: Videos play on any site, and usually inside an iframe from
  another domain (the player's host), so the content script has to run in every frame to
  find the video, and request observation has to cover the player's subtitle hosts.

**Remote code:** No, the extension does not use remote code.

**Data usage:** The extension handles website content and web history in the sense that it
reads the page title, address and video state, but sends them only to the user's own
computer (127.0.0.1). Nothing is transmitted to the developer or any third party. Tick:
"I do not sell or transfer user data to third parties", "I do not use or transfer user data
for purposes that are unrelated to my item's single purpose", "I do not use or transfer user
data to determine creditworthiness or for lending purposes".

**Privacy policy URL:** https://github.com/Nymbo/Live-Subtitles/blob/main/PRIVACY.md

**Homepage / support URL:** https://github.com/Nymbo/Live-Subtitles

## Assets

- Icon: `extension/icons/icon128.png` (in the zip; the dashboard also asks for it)
- Screenshots: `dist/store/screenshot-*.png` (1280×800)
- Small promo tile: `dist/store/promo-440x280.png`
