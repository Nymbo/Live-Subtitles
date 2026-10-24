"""Sound events for tracks without sound descriptions: AudioSet tagging with AST.

MIT's Audio Spectrogram Transformer fine-tuned on AudioSet (527 classes), from
onnx-community's quantized export (~90 MB, runs on the CPU in about a third of a second
per 10 s window). It knows sounds, not story: it can say [music] or [phone ringing],
never [ominous music]. Only a curated set of classes becomes a cue, each with its own
threshold, and only confident, sustained events.

Runs inside the media worker process (it needs ONNX Runtime). Features follow
ASTFeatureExtractor exactly: Kaldi fbank, 128 mel bins, 25 ms Hann windows every 10 ms,
1024 frames (10.24 s), normalised with AudioSet's mean and twice its std.
"""

from __future__ import annotations

import json
import math

import numpy as np

REPO = "onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX"
MODEL_FILE = "onnx/model_quantized.onnx"
SR = 16000
FRAMES = 1024
MEAN, STD = -4.2677393, 4.5689974
WINDOW_S = FRAMES * 0.01 + 0.015      # 10.24 s of frames plus one window length

# AudioSet label -> (caption, threshold). Speech, generic "Music" sub-genres and vague
# classes ("Inside, small room") are deliberately absent.
CAPTIONS: dict[str, tuple[str, float]] = {
    "Music": ("[music]", 0.55),
    "Singing": ("[singing]", 0.45),
    "Laughter": ("[laughter]", 0.35),
    "Giggle": ("[giggling]", 0.4),
    "Crying, sobbing": ("[crying]", 0.35),
    "Baby cry, infant cry": ("[baby crying]", 0.4),
    "Screaming": ("[screaming]", 0.35),
    "Applause": ("[applause]", 0.4),
    "Cheering": ("[cheering]", 0.4),
    "Crowd": ("[crowd murmuring]", 0.5),
    "Gunshot, gunfire": ("[gunshots]", 0.3),
    "Machine gun": ("[gunfire]", 0.3),
    "Explosion": ("[explosion]", 0.3),
    "Thunder": ("[thunder]", 0.35),
    "Rain": ("[rain]", 0.5),
    "Telephone bell ringing": ("[phone ringing]", 0.35),
    "Ringtone": ("[phone ringing]", 0.35),
    "Telephone": ("[phone ringing]", 0.5),
    "Doorbell": ("[doorbell]", 0.35),
    "Knock": ("[knocking]", 0.35),
    "Door": ("[door]", 0.55),
    "Slam": ("[door slams]", 0.4),
    "Dog": ("[dog barking]", 0.45),
    "Bark": ("[dog barking]", 0.4),
    "Cat": ("[cat meowing]", 0.45),
    "Siren": ("[siren]", 0.35),
    "Police car (siren)": ("[police siren]", 0.35),
    "Ambulance (siren)": ("[ambulance siren]", 0.35),
    "Car alarm": ("[car alarm]", 0.35),
    "Vehicle horn, car horn, honking": ("[car horn]", 0.4),
    "Glass": ("[glass shatters]", 0.5),
    "Shatter": ("[glass shatters]", 0.35),
    "Footsteps": ("[footsteps]", 0.45),
    "Whistling": ("[whistling]", 0.4),
    "Cough": ("[coughing]", 0.4),
    "Gasp": ("[gasps]", 0.4),
    "Sigh": ("[sighs]", 0.5),
    "Alarm clock": ("[alarm ringing]", 0.4),
    "Smoke detector, smoke alarm": ("[alarm beeping]", 0.4),
    "Helicopter": ("[helicopter]", 0.45),
    "Fireworks": ("[fireworks]", 0.4),
    "Wind": ("[wind howling]", 0.6),
}


def _mel_banks(num_bins: int = 128, n_fft: int = 512, sr: int = SR, low: float = 20.0) -> np.ndarray:
    """Kaldi's triangular mel filters (mel = 1127 ln(1 + f/700)), shape (n_fft//2 + 1, num_bins)."""
    high = sr / 2
    mel = lambda f: 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)  # noqa: E731
    lo, hi = mel(low), mel(high)
    delta = (hi - lo) / (num_bins + 1)
    bins = mel(np.arange(n_fft // 2) * sr / n_fft)
    banks = np.zeros((n_fft // 2 + 1, num_bins), np.float32)
    for m in range(num_bins):
        left, center, right = lo + m * delta, lo + (m + 1) * delta, lo + (m + 2) * delta
        up = (bins - left) / (center - left)
        down = (right - bins) / (right - center)
        banks[: n_fft // 2, m] = np.maximum(0.0, np.minimum(up, down))
    return banks


_BANKS = None


def fbank(wave: np.ndarray) -> np.ndarray:
    """torchaudio.compliance.kaldi.fbank(window_type='hanning', num_mel_bins=128), in numpy."""
    global _BANKS
    if _BANKS is None:
        _BANKS = _mel_banks()
    n = 1 + (len(wave) - 400) // 160 if len(wave) >= 400 else 0
    if n <= 0:
        return np.zeros((0, 128), np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(wave.astype(np.float64), 400)[::160][:n].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
    frames[:, 0] -= 0.97 * frames[:, 0]
    frames *= 0.5 - 0.5 * np.cos(2 * math.pi * np.arange(400) / 399)
    spec = np.abs(np.fft.rfft(frames, 512)) ** 2
    energies = spec @ _BANKS
    return np.log(np.maximum(energies, np.finfo(np.float32).eps)).astype(np.float32)


def features(wave: np.ndarray) -> np.ndarray:
    f = fbank(wave)[:FRAMES]
    if len(f) < FRAMES:
        f = np.pad(f, ((0, FRAMES - len(f)), (0, 0)))
    return ((f - MEAN) / (STD * 2)).astype(np.float32)


class AudioTagger:
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO, MODEL_FILE)
        config = json.loads(open(hf_hub_download(REPO, "config.json"), encoding="utf-8").read())
        self.labels = [config["id2label"][str(i)] for i in range(len(config["id2label"]))]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.watch = [(i, *CAPTIONS[l]) for i, l in enumerate(self.labels) if l in CAPTIONS]

    def probabilities(self, wave: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.input: features(wave)[None]})[0][0]
        return 1.0 / (1.0 + np.exp(-logits))

    def tag(self, wave: np.ndarray, t0: float, hop_s: float = 5.0) -> list[tuple[str, float, float, float]]:
        """Sound events in `wave` (16 kHz, starting at t0): [(caption, start, end, confidence)]."""
        win, hop = int(WINDOW_S * SR), int(hop_s * SR)
        events: list[tuple[str, float, float, float]] = []
        for i in range(0, max(1, len(wave) - win + hop), hop):
            piece = wave[i:i + win]
            if len(piece) < SR * 2:
                break
            probs = self.probabilities(piece)
            start, end = t0 + i / SR, t0 + (i + len(piece)) / SR
            best: dict[str, float] = {}
            for idx, caption, thr in self.watch:
                p = float(probs[idx])
                if p >= thr:
                    best[caption] = max(best.get(caption, 0.0), p)
            for caption, p in best.items():
                events.append((caption, start, end, p))
        return merge_events(events)


def merge_events(events: list[tuple[str, float, float, float]], gap_s: float = 1.0):
    """Join overlapping windows of the same caption into one event."""
    out: list[list] = []
    for caption, a, b, p in sorted(events, key=lambda e: (e[0], e[1])):
        if out and out[-1][0] == caption and a <= out[-1][2] + gap_s:
            out[-1][2] = max(out[-1][2], b)
            out[-1][3] = max(out[-1][3], p)
        else:
            out.append([caption, a, b, p])
    return [tuple(e) for e in sorted(out, key=lambda e: e[1])]
