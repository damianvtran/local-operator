"""Master the 10 picked notification clips into lossless reference files.

Why: the generated MP3s vary by ~12 LU in loudness, carry leading silence
(up to 120 ms, which reads as notification lag), and some have a noisy
tail floor (signal-handshake's audible "fry"). Masters are the single
source every per-platform encode is derived from, so they stay lossless
(44.1 kHz, the source rate, so no resampling; stereo is kept because
soft-pulse is wide and would hollow out folded to mono).

Steps per clip:
1. Decode to float.
2. Trim leading silence to 5 ms before the first 10 ms window within
   40 dB of the peak window.
3. Cut the tail where the 10 ms RMS stays 60 dB below the peak
   (a per-clip override for signal-handshake removes the fry), then apply
   a raised-cosine fade over the last FADE ms so no cut clicks.
4. Loudness-normalize to TARGET_LUFS integrated with a true-peak ceiling.
   Short clips gate poorly under BS.1770 integrated loudness, so we
   normalize on max momentary (400 ms) loudness instead: it tracks
   perceived "how loud is the ping".

Usage: python master.py <src_dir> <out_dir>
"""

import json
import os
import subprocess
import sys

import numpy as np

FF = "/opt/homebrew/bin/ffmpeg"
SR = 44100
TARGET_M = -20.0  # max momentary LUFS reference level; playback volume scales down from here
TP_CEIL = -3.0  # dBTP ceiling after normalization (leaves codec overshoot headroom)
FADE_MS = 40

# Per-clip overrides measured by hand (seconds, in SOURCE time).
OVERRIDES = {
    # Fry: broadband floor at about -69 dB from 0.66 s; HF ratio already climbs
    # at 0.62 s. Cut at 0.62 and fade from 0.56 so the decay ends clean.
    # Pre-onset noise 0.02-0.12 s goes with the leading trim.
    "06-signal-handshake": {"end": 0.62, "fade_ms": 60},
}

ID = {
    "01-two-note-chime": "two-note-chime",
    "02-felt-marimba": "felt-marimba",
    "03-hum-swell": "hum-swell",
    "04-task-complete": "task-complete",
    "05-soft-pulse": "soft-pulse",
    "06-signal-handshake": "signal-handshake",
    "07-tape-warble": "tape-warble",
    "08-bubble-chord": "bubble-chord",
    "09-airy-whistle": "airy-whistle",
    "10-bubble-pulse": "bubble-pulse",
}


def decode(path):
    raw = subprocess.run(
        [FF, "-v", "error", "-i", path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)


def rms_db(x, w):
    m = x.mean(1)
    return np.array(
        [
            20 * np.log10(np.sqrt((m[i : i + w] ** 2).mean()) + 1e-12)
            for i in range(0, len(m) - w, w)
        ]
    )


def measure(x):
    """Return (integrated LUFS, true peak dBTP, max momentary LUFS) via ffmpeg ebur128.

    The clip is padded with 0.5 s of silence on each side: BS.1770 momentary
    loudness uses a 400 ms window, and a clip shorter than that (bubble-pulse
    is ~0.3 s) otherwise yields no valid momentary block and a garbage gain.
    """
    pad = np.zeros((int(0.5 * SR), 2))
    x = np.concatenate([pad, x, pad])
    p = subprocess.run(
        [
            FF,
            "-hide_banner",
            "-nostats",
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            str(SR),
            "-i",
            "-",
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        input=x.astype(np.float32).tobytes(),
        capture_output=True,
    )
    err = p.stderr.decode()
    summ = err[err.rfind("Summary:") :]
    i_l = float(summ.split("I:")[1].split()[0])
    tp = float(summ.split("Peak:")[1].split()[0])
    mm = max(float(t.split()[0]) for t in err[: err.rfind("Summary:")].split(" M:")[1:])
    return i_l, tp, mm


def master(src, name):
    x = decode(src)
    w = int(0.01 * SR)
    r = rms_db(x, w)
    pk = r.max()
    on = max(0, int(np.argmax(r > pk - 40)) * w - int(0.005 * SR))
    ov = OVERRIDES.get(name, {})
    if "end" in ov:
        end = int(ov["end"] * SR)
    else:
        above = np.where(r > pk - 60)[0]
        end = min(len(x), (above[-1] + 1) * w + int(0.02 * SR))
    y = x[on:end].copy()
    n = int(ov.get("fade_ms", FADE_MS) / 1000 * SR)
    n = min(n, len(y) // 3)
    y[-n:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n)))[:, None]
    # tiny 2 ms fade-in so a trimmed onset never clicks
    k = int(0.002 * SR)
    y[:k] *= np.linspace(0, 1, k)[:, None]
    _, _, mm = measure(y)
    g = 10 ** ((TARGET_M - mm) / 20)
    y *= g
    _, tp, _ = measure(y)
    if tp > TP_CEIL:
        y *= 10 ** ((TP_CEIL - tp) / 20)
    return x, y, on / SR, end / SR


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    report = []
    for stem, sid in ID.items():
        x, y, t0, t1 = master(os.path.join(src, stem + ".mp3"), stem)
        before = measure(x)
        after = measure(y)
        dst = os.path.join(out, sid + ".wav")
        subprocess.run(
            [
                FF,
                "-v",
                "error",
                "-y",
                "-f",
                "f32le",
                "-ac",
                "2",
                "-ar",
                str(SR),
                "-i",
                "-",
                "-c:a",
                "pcm_s24le",
                dst,
            ],
            input=y.astype(np.float32).tobytes(),
            check=True,
        )
        report.append(
            {
                "id": sid,
                "src_len": round(len(x) / SR, 3),
                "trim": [round(t0, 3), round(t1, 3)],
                "len": round(len(y) / SR, 3),
                "before": before,
                "after": after,
            }
        )
    json.dump(report, open(os.path.join(out, "report.json"), "w"), indent=1)
    for r in report:
        print(
            f"{r['id']:17} {r['src_len']:.2f}s -> {r['len']:.2f}s trim {r['trim']}  "
            f"I/TP/Mmax {r['before'][0]:6.1f}/{r['before'][1]:5.1f}/{r['before'][2]:6.1f} -> "
            f"{r['after'][0]:6.1f}/{r['after'][1]:5.1f}/{r['after'][2]:6.1f}"
        )


if __name__ == "__main__":
    main()
