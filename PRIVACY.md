# Privacy policy

Live Subtitles is a Windows desktop app with an optional browser extension. This policy
covers both. Last updated 2026-10-24.

## The short version

Nothing you watch, and nothing about you, is sent to the developer or to anyone else by
the extension. The extension talks only to the Live Subtitles app on your own computer.

## What the extension does

While a page plays a video of two minutes or longer, the extension sends these details to
the Live Subtitles app at `http://127.0.0.1:47219` (your own computer, never the internet):

- the tab's title and address, the addresses of the page's frames, and the names the page
  gives what is playing (its heading, its metadata, the frames' titles);
- the video's position, length, playback rate, and whether it is paused;
- whether the tab is visible and making sound;
- the season and episode number if the page shows one (for example "S2 E3");
- the addresses of video streams and subtitle files the page's player requested.

The app uses them to work out what is playing, to line its subtitles up with the video, and
to use the page's own subtitle files when they exist. The extension has no server, no
analytics, no advertising, and no account. It stores one setting in the browser (the app's
port number).

## What the app sends over the internet

To find subtitles, the app sends the title, year, season and episode of what is playing to
the subtitle and title databases you use: IMDb's title suggestions, TMDB, OpenSubtitles,
Wyzie, SubDL and Gestdown (Addic7ed). It downloads subtitle files from them, and from the
page's own subtitle addresses. API keys you enter are stored encrypted for your Windows
account. Speech recognition and translation run on your computer; audio never leaves it.

## Contact

Questions: open an issue at https://github.com/Nymbo/Live-Subtitles/issues.
