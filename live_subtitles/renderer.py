"""Subtitle painting shared by the overlay and the settings preview.

Text is drawn as vector paths: soft shadow, then an outline stroked *outside* the glyphs
(stroke first, fill on top), then the fill. Boxes are unioned so stacked lines never
double their alpha where they overlap.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen

from .settings import Appearance


def color(value: str) -> QColor:
    c = QColor(value)
    return c if c.isValid() else QColor("#FFFFFFFF")


def make_font(ap: Appearance) -> QFont:
    font = QFont(ap.font_family)
    font.setPixelSize(max(8, int(ap.font_size)))
    font.setBold(ap.bold)
    font.setItalic(ap.italic)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


def effect_margin(ap: Appearance) -> float:
    """Room the outline and shadow need beyond the glyph box."""
    return ap.outline_width + max(0.0, ap.shadow_offset) + ap.shadow_softness + 2


def line_height(ap: Appearance) -> float:
    return QFontMetricsF(make_font(ap)).height() * ap.line_spacing


def block_height(ap: Appearance, lines: int) -> int:
    fm = QFontMetricsF(make_font(ap))
    return int(fm.height() + (lines - 1) * fm.height() * ap.line_spacing + 2 * ap.padding_y
               + 2 * effect_margin(ap) + 4)


def text_width_fn(ap: Appearance, width: float):
    """fits(text) for CaptionBuilder: does a line fit the overlay width in pixels?"""
    fm = QFontMetricsF(make_font(ap))
    avail = width - 2 * (ap.padding_x + effect_margin(ap))
    upper = ap.uppercase
    return lambda text: fm.horizontalAdvance(text.upper() if upper else text) <= avail


def paint_captions(p: QPainter, rect: QRectF, lines: list[str], ap: Appearance,
                   opacity: float = 1.0, scroll: float = 0.0, outgoing: str | None = None) -> None:
    """Draw `lines` bottom-anchored in `rect`.

    Roll-up animation: with `scroll` in (0, 1] every line sits that fraction of a line
    lower than its resting place, and `outgoing` (the line that just scrolled off) is drawn
    above the first line, fading with `scroll`. Animating scroll 1 -> 0 slides the block up.
    """
    if not lines:
        return
    if outgoing and scroll > 0:
        p.save()
        p.setClipRect(rect)
        _paint_lines(p, rect, [outgoing], ap, opacity * scroll * scroll, (scroll - 1.0) - (len(lines) - 1))
        p.restore()
    _paint_lines(p, rect, lines, ap, opacity, scroll)


def _paint_lines(p: QPainter, rect: QRectF, lines: list[str], ap: Appearance, opacity: float,
                 shift_lines: float) -> None:
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    p.setOpacity(opacity)
    font = make_font(ap)
    fm = QFontMetricsF(font)
    step = fm.height() * ap.line_spacing
    margin = effect_margin(ap)
    texts = [t.upper() if ap.uppercase else t for t in lines]

    # Bottom-anchored: the newest line sits on the same baseline as lines roll up.
    last_baseline = rect.bottom() - margin - ap.padding_y - fm.descent() + shift_lines * step
    left, right = rect.left() + margin + ap.padding_x, rect.right() - margin - ap.padding_x
    glyphs = QPainterPath()
    boxes: list[QRectF] = []
    for i, text in enumerate(texts):
        baseline = last_baseline - (len(texts) - 1 - i) * step
        if not text:          # a blank line keeps its slot but draws nothing (no empty box)
            continue
        w = fm.horizontalAdvance(text)
        if ap.align == "left":
            x = left
        elif ap.align == "right":
            x = right - w
        else:
            x = (left + right - w) / 2
        glyphs.addText(QPointF(x, baseline), font, text)
        top = baseline - fm.ascent()
        boxes.append(QRectF(x - ap.padding_x, top - ap.padding_y, w + 2 * ap.padding_x,
                            fm.height() + 2 * ap.padding_y))

    if ap.background in ("line", "block") and boxes:
        bg = QPainterPath()
        r = ap.corner_radius
        if ap.background == "block":
            union = boxes[0]
            for b in boxes[1:]:
                union = union.united(b)
            bg.addRoundedRect(union, r, r)
        else:
            for b in boxes:
                bg.addRoundedRect(b, r, r)
            bg = bg.simplified()
        p.fillPath(bg, color(ap.background_color))

    shadow = color(ap.shadow_color)
    if shadow.alpha() and (ap.shadow_offset or ap.shadow_softness):
        sp = glyphs.translated(ap.shadow_offset, ap.shadow_offset)
        if ap.shadow_softness > 0:
            # Approximate a blur with widening, fading strokes around the offset glyphs.
            steps = 5
            for k in range(steps, 0, -1):
                c = QColor(shadow)
                c.setAlphaF(shadow.alphaF() / (steps + 1))
                pen = QPen(c, (ap.outline_width * 2) + ap.shadow_softness * 2 * k / steps,
                           Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                p.strokePath(sp, pen)
        elif ap.outline_width > 0:
            p.strokePath(sp, QPen(shadow, ap.outline_width * 2, Qt.PenStyle.SolidLine,
                                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.fillPath(sp, shadow)

    if ap.outline_width > 0:
        p.strokePath(glyphs, QPen(color(ap.outline_color), ap.outline_width * 2, Qt.PenStyle.SolidLine,
                                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.fillPath(glyphs, color(ap.text_color))
    p.restore()
