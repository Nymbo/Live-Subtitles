"""Where the app keeps settings, models, and logs (outside the repo)."""

import os
from pathlib import Path

from . import APP_NAME


def _env_dir(var: str, fallback: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / fallback)


# Settings roam with the user profile; models and logs are machine-local and large.
CONFIG_DIR = _env_dir("APPDATA", "AppData/Roaming") / APP_NAME
DATA_DIR = _env_dir("LOCALAPPDATA", "AppData/Local") / APP_NAME
MODELS_DIR = DATA_DIR / "models"
LOG_DIR = DATA_DIR / "logs"
SETTINGS_FILE = CONFIG_DIR / "settings.json"


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, MODELS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
