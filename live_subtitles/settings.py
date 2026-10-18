"""User settings: dataclasses persisted as JSON, plus appearance presets.

Colors are "#AARRGGBB" strings (QColor's HexArgb form) so alpha travels with the color.
Loading tolerates missing and unknown keys, so old settings files keep working.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path

from .paths import SETTINGS_FILE

logger = logging.getLogger(__name__)


@dataclass
class Appearance:
    font_family: str = "Segoe UI"
    font_size: int = 40                 # px
    bold: bool = True
    italic: bool = False
    text_color: str = "#FFFFFFFF"
    outline_color: str = "#FF000000"
    outline_width: float = 3.0          # px, drawn outside the glyphs
    shadow_color: str = "#B4000000"
    shadow_offset: float = 2.0          # px, down-right
    shadow_softness: float = 4.0        # px of spread; 0 = hard shadow
    background: str = "none"            # none | line | block
    background_color: str = "#A0000000"
    padding_x: int = 14
    padding_y: int = 4
    corner_radius: int = 6
    line_spacing: float = 1.12
    align: str = "center"               # left | center | right
    uppercase: bool = False
    fade_ms: int = 180                  # fade-out when captions clear; 0 = instant


@dataclass
class Layout:
    max_lines: int = 2
    max_chars: int = 42                 # per line, the usual subtitle limit
    width: int = 1100                   # overlay width in px; text also wraps to it
    mode: str = "rollup"                # rollup: lines scroll up | pop: a full page clears
    clear_after_s: float = 4.0          # wipe captions after this much quiet
    sentence_breaks: bool = True        # start a new line after . ? !
    hide_fillers: bool = False          # drop "uh", "um", ...
    x: int | None = None                # top-left; None = bottom centre of the primary screen
    y: int | None = None
    locked: bool = False                # locked = click-through and not draggable
    visible: bool = True


@dataclass
class Engine:
    language: str = "en"                # "en" uses the English models; else a locale or "auto"
    # English: "auto" runs Parakeet-unified on a CUDA GPU and falls back to Nemotron on the
    # CPU when there is no usable GPU; "gpu" and "cpu" force one (other languages: CPU).
    recognizer: str = "auto"
    # Nemotron (CPU) chunk: 160 | 560 | 1120 ms. On the 16-scene benchmark 560 ms was as
    # accurate as 160 ms (17.0% vs 17.6% WER on real audio) at a third of the CPU; 160 ms
    # needed ~9 cores to stay ahead of real time.
    latency_ms: int = 560
    punctuation: bool = True            # Nemotron English: re-punctuate with a dedicated model
    threads: int = 2                    # Nemotron (CPU) threads: 560 ms needs ~1.9 cores at 2
    device: str = ""                    # loopback device name; "" = Windows default output
    dialogue_boost: bool = True         # favour the centre channel on 5.1/7.1 output
    autostart: bool = True              # start captioning when the app launches


@dataclass
class Hotkeys:
    toggle: str = "Ctrl+Alt+S"          # Ctrl+Alt+C is often taken (it was on the dev machine)
    lock: str = "Ctrl+Alt+L"
    visibility: str = "Ctrl+Alt+H"


@dataclass
class Files:
    formats: list[str] = field(default_factory=lambda: ["srt", "ass"])
    use_gpu: bool = True
    overwrite: bool = False
    max_line_chars: int = 42
    max_cue_lines: int = 2


@dataclass
class Settings:
    appearance: Appearance = field(default_factory=Appearance)
    layout: Layout = field(default_factory=Layout)
    engine: Engine = field(default_factory=Engine)
    hotkeys: Hotkeys = field(default_factory=Hotkeys)
    files: Files = field(default_factory=Files)
    custom_presets: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        return _build(cls, data)


def _build(cls, data):
    """Construct a dataclass from a dict, ignoring unknown keys and keeping defaults for missing ones."""
    if not isinstance(data, dict):
        return cls()
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        default = getattr(cls(), f.name)
        if is_dataclass(default):
            kwargs[f.name] = _build(type(default), value)
        elif isinstance(default, bool):
            kwargs[f.name] = bool(value)
        elif isinstance(default, (int, float)) and not isinstance(value, (int, float)):
            continue
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load(path: Path = SETTINGS_FILE) -> Settings:
    try:
        return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return Settings()
    except Exception:
        logger.exception("Could not read %s; using defaults", path)
        return Settings()


def save(settings: Settings, path: Path = SETTINGS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)


# Built-in looks. Each preset only lists what it changes from the Appearance defaults.
PRESETS: dict[str, dict] = {
    "Clean white": {},
    "Classic yellow": {"text_color": "#FFFFE14D", "outline_width": 2.5, "shadow_softness": 2.0},
    "Boxed": {"background": "line", "background_color": "#B8000000", "outline_width": 0.0,
              "shadow_offset": 0.0, "shadow_softness": 0.0, "bold": False, "padding_x": 12, "padding_y": 3},
    "Soft block": {"background": "block", "background_color": "#80101014", "outline_width": 0.0,
                   "shadow_offset": 1.0, "shadow_softness": 3.0, "corner_radius": 12, "bold": False,
                   "padding_x": 22, "padding_y": 10},
    "Cinema": {"font_family": "Georgia", "bold": False, "italic": False, "font_size": 42,
               "text_color": "#FFF5F1E6", "outline_width": 2.0, "shadow_offset": 3.0, "shadow_softness": 6.0},
    "High contrast": {"text_color": "#FFFFFF00", "background": "line", "background_color": "#FF000000",
                      "outline_width": 0.0, "shadow_offset": 0.0, "shadow_softness": 0.0, "font_size": 44},
    "Minimal": {"font_size": 34, "bold": False, "outline_width": 1.5, "shadow_offset": 0.0,
                "shadow_softness": 6.0, "shadow_color": "#C0000000"},
}


def preset_appearance(name: str, custom: dict[str, dict] | None = None) -> Appearance:
    overrides = (custom or {}).get(name, PRESETS.get(name, {}))
    return _build(Appearance, {**asdict(Appearance()), **overrides})


def appearance_overrides(appearance: Appearance) -> dict:
    """The keys where an appearance differs from the defaults (what a saved preset stores)."""
    base = asdict(Appearance())
    return {k: v for k, v in asdict(appearance).items() if base[k] != v}


def matching_preset(appearance: Appearance, custom: dict[str, dict] | None = None) -> str | None:
    for name in [*PRESETS, *(custom or {})]:
        if preset_appearance(name, custom) == appearance:
            return name
    return None


__all__ = ["Settings", "Appearance", "Layout", "Engine", "Hotkeys", "Files", "PRESETS",
           "load", "save", "preset_appearance", "appearance_overrides", "matching_preset", "replace"]
