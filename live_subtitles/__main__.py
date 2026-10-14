"""Entry point.

    python -m live_subtitles                  # tray app (live subtitles + settings)
    python -m live_subtitles --settings       # same, with the settings window open
    python -m live_subtitles transcribe FILE  # write subtitle files for videos, no GUI
"""

import sys


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "transcribe":
        import logging

        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
        from .transcribe import run

        return run(argv[1:])
    from .app import main as app_main

    return app_main(open_settings="--settings" in argv)


if __name__ == "__main__":
    sys.exit(main())
