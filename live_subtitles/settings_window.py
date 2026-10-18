"""Settings window. Every control applies immediately; the preview uses the overlay's renderer."""

from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFontComboBox, QFormLayout, QGroupBox,
    QHBoxLayout, QInputDialog, QKeySequenceEdit, QLabel, QMessageBox, QPushButton, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from . import models
from .captions import CaptionBuilder
from .renderer import block_height, paint_captions, text_width_fn
from .settings import (PRESETS, Appearance, Settings, appearance_overrides, matching_preset,
                       preset_appearance)

SAMPLE = ("The storm is getting closer, we should head back before it gets dark. "
          "Wait. Did you hear that? Something is moving out there.")


class ColorButton(QPushButton):
    changed = Signal(str)

    def __init__(self, value: str):
        super().__init__()
        self.setFixedWidth(150)
        self.clicked.connect(self._pick)
        self.set_value(value)

    def set_value(self, value: str) -> None:
        self.value = value
        c = QColor(value)
        fg = "#000" if c.lightness() > 140 and c.alpha() > 100 else "#fff"
        self.setText(f"{c.name(QColor.NameFormat.HexRgb).upper()}  {round(c.alphaF() * 100)}%")
        self.setStyleSheet(f"QPushButton {{ background: rgba({c.red()},{c.green()},{c.blue()},{c.alpha()});"
                           f" color: {fg}; border: 1px solid #888; border-radius: 4px; padding: 4px; }}")

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor(self.value), self, "Pick a colour",
                                  QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self.set_value(c.name(QColor.NameFormat.HexArgb).upper())
            self.changed.emit(self.value)


class Preview(QWidget):
    """A miniature of the overlay, drawn at the overlay's real width and scaled to fit."""

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.bright = False
        self.setMinimumHeight(170)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        g = QLinearGradient(0, 0, 0, r.height())
        if self.bright:
            g.setColorAt(0, QColor("#BFD9F2"))
            g.setColorAt(0.6, QColor("#F4E7C8"))
            g.setColorAt(1, QColor("#D9C59A"))
        else:
            g.setColorAt(0, QColor("#141A2B"))
            g.setColorAt(0.6, QColor("#3A2F44"))
            g.setColorAt(1, QColor("#1C1712"))
        p.fillRect(r, g)
        # A horizon and a few shapes so text is judged against a busy frame, not a flat one.
        hills = QPainterPath()
        hills.moveTo(0, r.height() * 0.62)
        hills.cubicTo(r.width() * 0.3, r.height() * 0.45, r.width() * 0.55, r.height() * 0.8, r.width(), r.height() * 0.58)
        hills.lineTo(r.width(), r.height())
        hills.lineTo(0, r.height())
        p.fillPath(hills, QColor("#E8E2D0" if self.bright else "#0B0D12"))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 200) if not self.bright else QColor(255, 250, 240, 255))
        p.drawEllipse(QRectF(r.width() * 0.78, r.height() * 0.12, 26, 26))

        s = self.settings
        ap, lay = s.appearance, s.layout
        height = block_height(ap, lay.max_lines)
        scale = min(1.0, (r.width() - 24) / max(1, lay.width), (r.height() - 16) / max(1, height))
        builder = CaptionBuilder(lay.max_lines, lay.max_chars, lay.mode, lay.sentence_breaks,
                                 fits=text_width_fn(ap, lay.width))
        builder.add(SAMPLE.split(), "", 0)
        p.save()
        p.translate((r.width() - lay.width * scale) / 2, r.height() - height * scale - 8)
        p.scale(scale, scale)
        paint_captions(p, QRectF(0, 0, lay.width, height), builder.display_lines(), ap)
        p.restore()
        p.end()


class SettingsWindow(QWidget):
    changed = Signal(str)               # appearance | layout | engine | hotkeys | files
    snap_requested = Signal()
    transcribe_requested = Signal()

    def __init__(self, settings: Settings, devices_fn):
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle("Live-Subtitles settings")
        self.resize(760, 820)
        self.s = settings
        self.devices_fn = devices_fn
        self._loading = False
        self.widgets: dict[str, QWidget] = {}

        root = QVBoxLayout(self)
        self.preview = Preview(settings)
        root.addWidget(self.preview, 1)
        bar = QHBoxLayout()
        bright = QCheckBox("Bright scene")
        bright.toggled.connect(self._toggle_bright)
        bar.addWidget(bright)
        bar.addStretch()
        bar.addWidget(QLabel("Changes apply right away and are saved automatically."))
        root.addLayout(bar)

        tabs = QTabWidget()
        tabs.addTab(self._look_tab(), "Look")
        tabs.addTab(self._captions_tab(), "Captions")
        tabs.addTab(self._engine_tab(), "Audio && model")
        tabs.addTab(self._hotkeys_tab(), "Hotkeys")
        tabs.addTab(self._files_tab(), "Video files")
        root.addWidget(tabs, 3)
        self.refresh()

    # -- helpers ---------------------------------------------------------------------
    def _bind(self, key: str, widget: QWidget, section: str, attr: str) -> QWidget:
        """Wire a control to settings.<section>.<attr>."""
        self.widgets[key] = widget
        target = lambda: getattr(self.s, section)  # noqa: E731

        def store(value):
            if self._loading:
                return
            setattr(target(), attr, value)
            if section == "appearance":
                self._sync_preset_combo()
            self.preview.update()
            self.changed.emit(section)

        if isinstance(widget, QCheckBox):
            widget.toggled.connect(store)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(store)
        elif isinstance(widget, QFontComboBox):
            widget.currentFontChanged.connect(lambda f: store(f.family()))
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(lambda _i: store(widget.currentData()))
        elif isinstance(widget, ColorButton):
            widget.changed.connect(store)
        widget.setProperty("binding", (section, attr))
        return widget

    @staticmethod
    def _spin(lo, hi, step=1, decimals=None, suffix=""):
        if decimals is None:
            w = QSpinBox()
        else:
            w = QDoubleSpinBox()
            w.setDecimals(decimals)
        w.setRange(lo, hi)
        w.setSingleStep(step)
        if suffix:
            w.setSuffix(suffix)
        w.setKeyboardTracking(False)
        return w

    @staticmethod
    def _combo(items: list[tuple[str, object]]) -> QComboBox:
        w = QComboBox()
        for label, data in items:
            w.addItem(label, data)
        return w

    def _toggle_bright(self, on: bool) -> None:
        self.preview.bright = on
        self.preview.update()

    # -- tabs ------------------------------------------------------------------------
    def _look_tab(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)

        presets = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(200)
        self.preset_combo.activated.connect(self._apply_preset)
        save = QPushButton("Save as preset…")
        save.clicked.connect(self._save_preset)
        self.delete_preset = QPushButton("Delete preset")
        self.delete_preset.clicked.connect(self._delete_preset)
        presets.addWidget(QLabel("Preset"))
        presets.addWidget(self.preset_combo, 1)
        presets.addWidget(save)
        presets.addWidget(self.delete_preset)
        col.addLayout(presets)

        text = QGroupBox("Text")
        f = QFormLayout(text)
        f.addRow("Font", self._bind("font", QFontComboBox(), "appearance", "font_family"))
        f.addRow("Size", self._bind("size", self._spin(12, 160, suffix=" px"), "appearance", "font_size"))
        style = QHBoxLayout()
        for key, label, attr in (("bold", "Bold", "bold"), ("italic", "Italic", "italic"),
                                 ("upper", "UPPERCASE", "uppercase")):
            style.addWidget(self._bind(key, QCheckBox(label), "appearance", attr))
        style.addStretch()
        f.addRow("Style", style)
        f.addRow("Colour", self._bind("text_color", ColorButton("#FFFFFFFF"), "appearance", "text_color"))
        f.addRow("Alignment", self._bind("align", self._combo(
            [("Centre", "center"), ("Left", "left"), ("Right", "right")]), "appearance", "align"))
        f.addRow("Line spacing", self._bind("spacing", self._spin(0.8, 2.5, 0.05, 2, "×"), "appearance", "line_spacing"))
        col.addWidget(text)

        edge = QGroupBox("Outline and shadow")
        f = QFormLayout(edge)
        f.addRow("Outline width", self._bind("outline_w", self._spin(0, 12, 0.5, 1, " px"), "appearance", "outline_width"))
        f.addRow("Outline colour", self._bind("outline_c", ColorButton("#FF000000"), "appearance", "outline_color"))
        f.addRow("Shadow distance", self._bind("shadow_o", self._spin(0, 12, 0.5, 1, " px"), "appearance", "shadow_offset"))
        f.addRow("Shadow softness", self._bind("shadow_s", self._spin(0, 20, 0.5, 1, " px"), "appearance", "shadow_softness"))
        f.addRow("Shadow colour", self._bind("shadow_c", ColorButton("#FF000000"), "appearance", "shadow_color"))
        col.addWidget(edge)

        box = QGroupBox("Background")
        f = QFormLayout(box)
        f.addRow("Box", self._bind("bg", self._combo(
            [("None", "none"), ("Behind each line", "line"), ("One box behind all lines", "block")]),
            "appearance", "background"))
        f.addRow("Box colour", self._bind("bg_c", ColorButton("#A0000000"), "appearance", "background_color"))
        pad = QHBoxLayout()
        pad.addWidget(self._bind("pad_x", self._spin(0, 60, suffix=" px wide"), "appearance", "padding_x"))
        pad.addWidget(self._bind("pad_y", self._spin(0, 40, suffix=" px tall"), "appearance", "padding_y"))
        f.addRow("Padding", pad)
        f.addRow("Corner radius", self._bind("radius", self._spin(0, 40, suffix=" px"), "appearance", "corner_radius"))
        f.addRow("Fade out", self._bind("fade", self._spin(0, 2000, 50, suffix=" ms"), "appearance", "fade_ms"))
        col.addWidget(box)
        col.addStretch()
        return page

    def _captions_tab(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        g = QGroupBox("Lines")
        f = QFormLayout(g)
        f.addRow("Lines on screen", self._bind("max_lines", self._spin(1, 4), "layout", "max_lines"))
        f.addRow("Characters per line", self._bind("max_chars", self._spin(16, 90), "layout", "max_chars"))
        f.addRow("Overlay width", self._bind("width", self._spin(240, 3840, 10, suffix=" px"), "layout", "width"))
        f.addRow("When lines fill", self._bind("mode", self._combo(
            [("Roll up (the top line scrolls away)", "rollup"),
             ("Pop on (clear and start a fresh block)", "pop")]), "layout", "mode"))
        f.addRow("Clear after quiet", self._bind("clear", self._spin(1, 30, 0.5, 1, " s"), "layout", "clear_after_s"))
        f.addRow("", self._bind("sentence", QCheckBox("Start a new line after each sentence"), "layout", "sentence_breaks"))
        f.addRow("", self._bind("fillers", QCheckBox("Hide filler words (uh, um, er…)"), "layout", "hide_fillers"))
        col.addWidget(g)

        g = QGroupBox("Overlay window")
        f = QFormLayout(g)
        f.addRow("", self._bind("visible", QCheckBox("Show subtitles"), "layout", "visible"))
        f.addRow("", self._bind("locked", QCheckBox("Lock in place (click-through)"), "layout", "locked"))
        snap = QPushButton("Snap to bottom centre")
        snap.clicked.connect(self.snap_requested.emit)
        f.addRow("", snap)
        tip = QLabel("Unlocked, the overlay can be dragged anywhere, resized from either side, and "
                     "Ctrl+scrolled to change the text size. Double-click it to re-centre it.")
        tip.setWordWrap(True)
        f.addRow(tip)
        col.addWidget(g)
        col.addStretch()
        return page

    def _engine_tab(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        g = QGroupBox("Audio")
        f = QFormLayout(g)
        row = QHBoxLayout()
        self.device_combo = QComboBox()
        self._bind("device", self.device_combo, "engine", "device")
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._fill_devices)
        row.addWidget(self.device_combo, 1)
        row.addWidget(refresh)
        f.addRow("Listen to", row)
        f.addRow("", self._bind("boost", QCheckBox("Favour the dialogue channel on 5.1/7.1 output"),
                                "engine", "dialogue_boost"))
        col.addWidget(g)

        g = QGroupBox("Speech model")
        f = QFormLayout(g)
        f.addRow("Language", self._bind("language", self._combo(
            [(label, code) for code, label in models.LANGUAGES.items()]), "engine", "language"))
        f.addRow("Recogniser", self._bind("recognizer", self._combo(
            [("Automatic: GPU when available, else CPU", "auto"),
             ("GPU: Parakeet-unified (NVIDIA CUDA)", "gpu"),
             ("CPU: Nemotron streaming", "cpu")]), "engine", "recognizer"))
        f.addRow("CPU latency", self._bind("latency", self._combo(
            [("Balanced: words ~0.5 s after speech", 560),
             ("Fastest: ~0.3 s, needs a strong CPU", 160),
             ("Lightest: ~1 s", 1120)]), "engine", "latency_ms"))
        f.addRow("", self._bind("punct", QCheckBox("Better punctuation and capitals on the CPU (31 MB model)"),
                                "engine", "punctuation"))
        f.addRow("CPU threads", self._bind("threads", self._spin(1, 16), "engine", "threads"))
        f.addRow("", self._bind("autostart", QCheckBox("Start live subtitles when the app starts"),
                                "engine", "autostart"))
        self.model_status = QLabel()
        self.model_status.setWordWrap(True)
        f.addRow(self.model_status)
        note = QLabel("English runs NVIDIA Parakeet-unified on an NVIDIA GPU (2.5 GB download, most "
                      "accurate, light on the CPU). Without one, and for other languages, captions run "
                      "on the CPU with NVIDIA's Nemotron streaming models (~470 MB per language and "
                      "latency). Models download the first time they are used.")
        note.setWordWrap(True)
        f.addRow(note)
        col.addWidget(g)
        col.addStretch()
        return page

    def _hotkeys_tab(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        g = QGroupBox("Global hotkeys (work from any app, including fullscreen video)")
        f = QFormLayout(g)
        self.key_edits: dict[str, QKeySequenceEdit] = {}
        for attr, label in (("toggle", "Start / stop live subtitles"), ("lock", "Lock / unlock overlay"),
                            ("visibility", "Show / hide subtitles")):
            edit = QKeySequenceEdit()
            edit.setMaximumSequenceLength(1)
            edit.editingFinished.connect(lambda a=attr: self._store_hotkey(a))
            clear = QPushButton("Clear")
            clear.clicked.connect(lambda _=False, e=edit, a=attr: (e.clear(), self._store_hotkey(a)))
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(clear)
            f.addRow(label, row)
            self.key_edits[attr] = edit
        self.hotkey_status = QLabel()
        self.hotkey_status.setWordWrap(True)
        f.addRow(self.hotkey_status)
        col.addWidget(g)
        col.addStretch()
        return page

    def _files_tab(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        g = QGroupBox("Subtitle files for videos")
        f = QFormLayout(g)
        fmts = QHBoxLayout()
        self.format_checks: dict[str, QCheckBox] = {}
        for fmt, label in (("srt", "SRT (plays everywhere)"), ("ass", "ASS (keeps your style in VLC)"),
                           ("vtt", "WebVTT")):
            cb = QCheckBox(label)
            cb.toggled.connect(self._store_formats)
            self.format_checks[fmt] = cb
            fmts.addWidget(cb)
        fmts.addStretch()
        f.addRow("Write", fmts)
        f.addRow("Characters per line", self._bind("f_chars", self._spin(20, 80), "files", "max_line_chars"))
        f.addRow("Lines per subtitle", self._bind("f_lines", self._spin(1, 3), "files", "max_cue_lines"))
        f.addRow("", self._bind("f_gpu", QCheckBox("Use the GPU (much faster)"), "files", "use_gpu"))
        f.addRow("", self._bind("f_over", QCheckBox("Overwrite existing subtitle files"), "files", "overwrite"))
        note = QLabel("Files are written next to the video as <name>.srt, which VLC loads automatically. "
                      "If one already exists and overwrite is off, the new file is named "
                      "<name>.live-subtitles.srt instead. Video files use Parakeet TDT 0.6B v3, which "
                      "sees the whole scene at once and is more accurate than live mode.")
        note.setWordWrap(True)
        f.addRow(note)
        go = QPushButton("Create subtitles for a video…")
        go.clicked.connect(self.transcribe_requested.emit)
        f.addRow(go)
        col.addWidget(g)
        col.addStretch()
        return page

    # -- presets / special stores ------------------------------------------------------
    def _sync_preset_combo(self) -> None:
        name = matching_preset(self.s.appearance, self.s.custom_presets)
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for n in PRESETS:
            self.preset_combo.addItem(n, n)
        for n in self.s.custom_presets:
            self.preset_combo.addItem(f"{n} (yours)", n)
        if name is None:
            self.preset_combo.addItem("Custom", None)
            self.preset_combo.setCurrentIndex(self.preset_combo.count() - 1)
        else:
            self.preset_combo.setCurrentIndex(self.preset_combo.findData(name))
        self.preset_combo.blockSignals(False)
        self.delete_preset.setEnabled(name in self.s.custom_presets)

    def _apply_preset(self, _index: int) -> None:
        name = self.preset_combo.currentData()
        if name is None:
            return
        self.s.appearance = preset_appearance(name, self.s.custom_presets)
        self.refresh()
        self.changed.emit("appearance")

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        name = name.strip()
        if not ok or not name:
            return
        if name in PRESETS:
            QMessageBox.warning(self, "Save preset", "That name belongs to a built-in preset.")
            return
        self.s.custom_presets[name] = appearance_overrides(self.s.appearance)
        self._sync_preset_combo()
        self.changed.emit("appearance")

    def _delete_preset(self) -> None:
        name = self.preset_combo.currentData()
        if name in self.s.custom_presets:
            del self.s.custom_presets[name]
            self._sync_preset_combo()
            self.changed.emit("appearance")

    def _store_hotkey(self, attr: str) -> None:
        text = self.key_edits[attr].keySequence().toString(QKeySequence.SequenceFormat.PortableText)
        setattr(self.s.hotkeys, attr, text)
        self.changed.emit("hotkeys")

    def _store_formats(self) -> None:
        if self._loading:
            return
        self.s.files.formats = [f for f, cb in self.format_checks.items() if cb.isChecked()] or ["srt"]
        self.changed.emit("files")

    def _fill_devices(self) -> None:
        current = self.s.engine.device
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("Windows default output", "")
        try:
            names = self.devices_fn()
        except Exception:
            names = []
        for name in names:
            self.device_combo.addItem(name, name)
        if current and self.device_combo.findData(current) < 0:
            self.device_combo.addItem(f"{current} (not connected)", current)
        self.device_combo.setCurrentIndex(max(0, self.device_combo.findData(current)))
        self.device_combo.blockSignals(False)

    # -- refresh from settings -------------------------------------------------------
    def set_hotkey_failures(self, failed: list[str]) -> None:
        names = {"toggle": "start/stop", "lock": "lock", "visibility": "show/hide"}
        self.hotkey_status.setText("" if not failed else
                                   "Could not register: " + ", ".join(names.get(f, f) for f in failed)
                                   + ". Another program may already use that combination.")

    def set_model_status(self, text: str) -> None:
        self.model_status.setText(text)

    def refresh(self) -> None:
        """Load every control from the settings object (e.g. after a preset or overlay drag)."""
        self._loading = True
        try:
            for w in self.widgets.values():
                section, attr = w.property("binding")
                value = getattr(getattr(self.s, section), attr)
                w.blockSignals(True)
                if isinstance(w, QCheckBox):
                    w.setChecked(bool(value))
                elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                    w.setValue(value)
                elif isinstance(w, QFontComboBox):
                    w.setCurrentFont(QFont(value))
                elif isinstance(w, QComboBox) and w is not self.device_combo:
                    w.setCurrentIndex(max(0, w.findData(value)))
                elif isinstance(w, ColorButton):
                    w.set_value(value)
                w.blockSignals(False)
            self._fill_devices()
            for attr, edit in self.key_edits.items():
                edit.setKeySequence(QKeySequence(getattr(self.s.hotkeys, attr)))
            for fmt, cb in self.format_checks.items():
                cb.setChecked(fmt in self.s.files.formats)
            self._sync_preset_combo()
        finally:
            self._loading = False
        self.preview.update()


__all__ = ["SettingsWindow", "Appearance", "asdict"]
