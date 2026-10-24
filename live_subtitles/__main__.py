"""Entry point.

    python -m live_subtitles                  # tray app (live subtitles + settings)
    python -m live_subtitles --settings       # same, with the settings window open
    python -m live_subtitles transcribe FILE  # write subtitle files for videos, no GUI
    python -m live_subtitles unified-worker   # GPU live recogniser (started by the app)
    python -m live_subtitles media-worker     # media mode's speech models (started by the app)
    python -m live_subtitles media-generate SOURCE  # subtitles from a file or stream, as JSON lines
"""

import sys


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "unified-worker":
        import logging

        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
        from .unified import worker_main

        return worker_main(argv[1:])
    if argv and argv[0] == "media-worker":
        import logging

        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
        from .media.worker import worker_main as media_worker

        return media_worker(argv[1:])
    if argv and argv[0] == "media-generate":
        from .media.generate import main as generate_main

        return generate_main(argv[1:])
    if argv and argv[0] == "transcribe":
        import logging

        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
        from .transcribe import run

        return run(argv[1:])
    from .app import main as app_main

    return app_main(open_settings="--settings" in argv)


if __name__ == "__main__":
    sys.exit(main())
