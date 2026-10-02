#!/usr/bin/env python3
"""Build the PyroSight voice clips and the "audio" flash partition image.

Phrase ids and texts are parsed from core/src/ps_alerts.c (ps_phrase_text[]),
so the clip index always matches ps_phrase_t. Each phrase is synthesised with
espeak-ng when it is installed, otherwise a placeholder tone (distinct pitch per
phrase) is written so the firmware pipeline can be exercised end to end.

Outputs (in --out, default build/audio):
    <NN>_<NAME>.pcm   16 kHz mono signed 16-bit little-endian
    <NN>_<NAME>.wav   same audio, for listening
    audio.bin         partition image, format documented in
                      firmware/components/audio/include/audio_pack.h

Flash it with tools/flash_partitions.sh (or esptool write_flash <audio offset> audio.bin).

Recorded human clips are better than TTS in a noisy fireground: drop
16 kHz mono WAVs named <NAME>.wav (e.g. WAY_OUT_IS.wav) into --override-dir and
they replace the synthesised ones.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
import zlib

RATE = 16000
MAGIC = b"PSAU"
VERSION = 1
HDR_BYTES = 24
ENTRY_BYTES = 8
DEFAULT_PART_SIZE = 0x200000  # must match firmware/partitions.csv "audio"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_phrases(alerts_c: str, alerts_h: str) -> list[tuple[str, str]]:
    """Return [(NAME, text)] indexed by ps_phrase_t value."""
    with open(alerts_h, encoding="utf-8") as f:
        h = f.read()
    m = re.search(r"typedef enum\s*\{(.*?)\}\s*ps_phrase_t;", h, re.S)
    if not m:
        sys.exit("ps_phrase_t enum not found in " + alerts_h)
    names = []
    for line in m.group(1).splitlines():
        line = re.sub(r"/\*.*?\*/", "", line).strip()
        mm = re.match(r"PS_PHRASE_([A-Z0-9_]+)\s*(=\s*(\d+))?\s*,?", line)
        if not mm:
            continue
        if mm.group(3) is not None and int(mm.group(3)) != len(names):
            sys.exit("non-sequential ps_phrase_t values are not supported")
        names.append(mm.group(1))
    if names and names[-1] == "COUNT":
        names.pop()

    with open(alerts_c, encoding="utf-8") as f:
        c = f.read()
    m = re.search(r"ps_phrase_text\s*\[[^\]]*\]\s*=\s*\{(.*?)\};", c, re.S)
    if not m:
        sys.exit("ps_phrase_text[] not found in " + alerts_c)
    texts = {}
    for mm in re.finditer(r"\[\s*PS_PHRASE_([A-Z0-9_]+)\s*\]\s*=\s*\"((?:[^\"\\]|\\.)*)\"", m.group(1)):
        texts[mm.group(1)] = mm.group(2)
    return [(n, texts.get(n, "")) for n in names]


def read_wav_mono16(path: str) -> list[int]:
    with wave.open(path, "rb") as w:
        ch, sw, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        sys.exit(f"{path}: only 16-bit WAV supported")
    s = struct.unpack("<%dh" % (len(raw) // 2), raw)
    if ch > 1:
        s = [sum(s[i:i + ch]) // ch for i in range(0, len(s), ch)]
    return resample(list(s), rate, RATE)


def resample(x: list[int], src: int, dst: int) -> list[int]:
    if src == dst or not x:
        return x
    n_out = int(len(x) * dst / src)
    out = []
    for i in range(n_out):
        t = i * src / dst
        j = int(t)
        f = t - j
        a = x[j]
        b = x[j + 1] if j + 1 < len(x) else a
        out.append(int(round(a + (b - a) * f)))
    return out


def trim_and_normalise(s: list[int], peak: float = 0.89) -> list[int]:
    thr = 300
    i0 = next((i for i, v in enumerate(s) if abs(v) > thr), 0)
    i1 = next((i for i in range(len(s) - 1, -1, -1) if abs(s[i]) > thr), len(s) - 1)
    pad = RATE // 50
    s = s[max(0, i0 - pad):min(len(s), i1 + pad)]
    m = max((abs(v) for v in s), default=0)
    if m == 0:
        return s
    g = peak * 32767 / m
    return [max(-32768, min(32767, int(v * g))) for v in s]


def espeak(text: str, exe: str, voice: str, speed: int) -> list[int] | None:
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.wav")
        try:
            subprocess.run([exe, "-v", voice, "-s", str(speed), "-w", p, text], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return None
        return trim_and_normalise(read_wav_mono16(p))


def placeholder(idx: int) -> list[int]:
    """A distinct two-tone chirp per phrase so placeholders can be told apart."""
    f0 = 400 + 60 * idx
    out = []
    for k, (f, ms) in enumerate(((f0, 180), (0, 40), (f0 * 1.25, 180))):
        n = RATE * ms // 1000
        ramp = RATE // 200
        for i in range(n):
            env = min(1.0, i / ramp, (n - i) / ramp)
            out.append(int(0.5 * 32767 * env * math.sin(2 * math.pi * f * i / RATE)) if f else 0)
    return out


def write_wav(path: str, s: list[int]) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(struct.pack("<%dh" % len(s), *s))


def build_image(clips: list[bytes | None]) -> bytes:
    """Pack clips (index i = ps_phrase_t i, None/empty = missing)."""
    count = len(clips)
    idx_end = HDR_BYTES + count * ENTRY_BYTES
    off = (idx_end + 3) & ~3
    entries, data = [], bytearray()
    for c in clips:
        if not c:
            entries.append((0, 0))
            continue
        pad = (-(off + len(data))) & 3
        data += b"\0" * pad
        entries.append((off + len(data), len(c)))
        data += c
    index = b"".join(struct.pack("<II", o, n) for o, n in entries)
    image_size = off + len(data)
    hdr = MAGIC + struct.pack("<HHIBBHII", VERSION, count, RATE, 16, 1, 0, image_size,
                              zlib.crc32(index) & 0xFFFFFFFF)
    assert len(hdr) == HDR_BYTES
    return hdr + index + b"\0" * (off - idx_end) + bytes(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "audio"))
    ap.add_argument("--alerts-c", default=os.path.join(ROOT, "core", "src", "ps_alerts.c"))
    ap.add_argument("--alerts-h", default=os.path.join(ROOT, "core", "include", "pyrosight", "ps_alerts.h"))
    ap.add_argument("--placeholder", action="store_true", help="never use espeak-ng")
    ap.add_argument("--voice", default="en-us")
    ap.add_argument("--speed", type=int, default=165, help="espeak-ng words per minute")
    ap.add_argument("--override-dir", help="directory of recorded <NAME>.wav clips")
    ap.add_argument("--partition-size", type=lambda v: int(v, 0), default=DEFAULT_PART_SIZE)
    a = ap.parse_args()

    phrases = parse_phrases(a.alerts_c, a.alerts_h)
    exe = None if a.placeholder else (shutil.which("espeak-ng") or shutil.which("espeak"))
    if not exe and not a.placeholder:
        print("espeak-ng not found: writing placeholder tones", file=sys.stderr)
    os.makedirs(a.out, exist_ok=True)

    clips: list[bytes | None] = []
    for i, (name, text) in enumerate(phrases):
        if name == "NONE" or not text:
            clips.append(None)
            continue
        s = None
        src = "placeholder"
        if a.override_dir:
            p = os.path.join(a.override_dir, name + ".wav")
            if os.path.exists(p):
                s, src = trim_and_normalise(read_wav_mono16(p)), "recorded"
        if s is None and exe:
            s = espeak(text, exe, a.voice, a.speed)
            src = "espeak" if s else src
        if not s:
            s = placeholder(i)
        pcm = struct.pack("<%dh" % len(s), *s)
        base = os.path.join(a.out, f"{i:02d}_{name}")
        with open(base + ".pcm", "wb") as f:
            f.write(pcm)
        write_wav(base + ".wav", s)
        clips.append(pcm)
        print(f"{i:2d} {name:<18} {len(s) / RATE:5.2f} s  {src:<11} {text!r}")

    img = build_image(clips)
    if len(img) > a.partition_size:
        sys.exit(f"audio image {len(img)} B exceeds partition size {a.partition_size} B")
    out = os.path.join(a.out, "audio.bin")
    with open(out, "wb") as f:
        f.write(img)
    print(f"wrote {out}: {len(img)} bytes, {sum(1 for c in clips if c)} clips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
