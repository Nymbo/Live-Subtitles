"""Media subtitles: pre-timed English subtitles for known movies and TV, synced from the audio.

Live captions transcribe whatever plays. Media mode instead finds the title being watched
(from the browser window title, or the optional browser extension), fetches an existing
subtitle track, and uses speech recognition only as a sensor: words heard in the audio
are matched against the track's lines to work out where playback is, and the track's
whole lines are shown at their own start times, ahead of the speech.

Pure-Python parts (no Qt, no ONNX Runtime): `subfile` (parse SRT/VTT/ASS, sound
descriptions), `titles` (window titles -> queries), `align` (anchors -> time map),
`clock` (audio clock -> playhead), `sources` / `tmdb` (web APIs), `store` (SQLite cache).
`worker` runs the models in a child process; `controller` ties it together in the app.
"""
