"""Build the audio suites for the Whisper hallucination experiment (CPU only, ~2-4 min).

Reads only data_audio/raw/ (decoded with soundfile; no ffmpeg/torchcodec) and writes 16 kHz mono float32 clips to
data_audio/derived/<suite>.npy (all clips concatenated) plus <suite>.json manifests (id, offset, n, seconds, category,
source details, ref text for speech). Fixed seeds; rerunning gives identical files.

Suites
  held-out (never shown to the agent)
    nonspeech           300 clips, 5-30 s: silence 30, noise 60, environmental 140 (ESC-50 folds 3-5), music 70
    padded              300 LibriSpeech test-clean utterances with 5-10 s of silence / noise / music at both ends
    wer                 500 other LibriSpeech test-clean utterances
  practice (agent-visible; disjoint sources)
    practice_nonspeech  100 clips: silence 10, noise 20, environmental 45 (ESC-50 folds 1-2), music 25 (other tracks)
    practice_speech     100 LibriSpeech dev-clean utterances
    practice_padded      50 other dev-clean utterances padded with practice pads (other noise seeds, practice tracks)
    practice_bg          50 other dev-clean utterances mixed with practice ESC-50 sounds / music at SNR 0/5/10 dB
  hard speech guards (held-out)
    bg                  300 test-clean utterances (disjoint from padded / wer) mixed with held-out ESC-50 (folds 3-5)
                        or held-out music over the whole utterance at SNR 0 / 5 / 10 dB
    other               300 LibriSpeech test-other utterances (<= 29.5 s)

Choices (also written to data_audio/derived/prepare_log.json)
  - silence = digital zeros (6 held-out / 2 practice; every length gives the same Whisper input, because Whisper
    zero-pads to 30 s) plus near-silence (white or pink noise at -80 / -70 dBFS RMS).
  - noise = white and pink noise at -50 / -35 / -20 dBFS RMS.
  - ESC-50: classes crying_baby and laughing are excluded (voiced, speech-like human vocalisations); coughing,
    sneezing, snoring and breathing are kept but flagged human_vocal=True. An environmental clip is 1-6 ESC-50
    recordings of the same class concatenated (5-30 s).
  - music = lewtun/music_genres_small (FMA tracks, 30 s), genres "Instrumental" and "Chiptune / Glitch" only, to
    avoid sung lyrics (a transcript of lyrics would not be a hallucination). Tracks are split between practice and
    held-out; one excerpt (5-29 s) per track. Track 10150 is excluded (it has sung lyrics).
"""
import csv
import io
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from common import DERIVED, RAW, SR

ESC_DIR = RAW / "esc50" / "ESC-50-master"
LIBRI = RAW / "librispeech" / "LibriSpeech"
MUSIC_PARQUET = next((RAW / "music" / "lewtun_music_genres_small" / "data").glob("*.parquet"), None)
MUSIC_GENRES = ("Instrumental", "Chiptune / Glitch")
ESC_EXCLUDE = ("crying_baby", "laughing")
ESC_HUMAN_VOCAL = ("coughing", "sneezing", "snoring", "breathing")
EXCLUDE_SONGS = {10150: "FMA 'Instrumental' track with sung lyrics (Whisper transcribed them); was practice pns_music_010"}
BG_SNRS = (0, 5, 10)


# ----------------------------------------------------------------------------------------------- audio utils
def to16k(x, sr):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(1)
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x.astype(np.float32)


def read(path_or_bytes):
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes)
    x, sr = sf.read(src, dtype="float32", always_2d=False)
    return to16k(x, sr)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


def set_level(x, db):
    y = x * (10 ** (db / 20) / (np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))
    peak = np.abs(y).max()
    return (y / peak * 0.99 if peak > 0.99 else y).astype(np.float32)


def white(n, rng):
    return rng.standard_normal(n).astype(np.float32)


def pink(n, rng):
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    return np.fft.irfft(spec, n).astype(np.float32)


def noise(kind, n, rng, db):
    return set_level(white(n, rng) if kind == "white" else pink(n, rng), db)


# ----------------------------------------------------------------------------------------------- sources
def esc_rows():
    rows = list(csv.DictReader(open(ESC_DIR / "meta" / "esc50.csv", encoding="utf-8")))
    return [r for r in rows if r["category"] not in ESC_EXCLUDE]


def music_tracks():
    import pyarrow.parquet as pq
    t = pq.read_table(MUSIC_PARQUET, columns=["song_id", "genre"]).to_pylist()
    keep = [i for i, r in enumerate(t) if r["genre"] in MUSIC_GENRES]
    return t, keep


def music_audio(row_indices):
    import pyarrow.parquet as pq
    pf = pq.read_table(MUSIC_PARQUET, columns=["audio", "song_id", "genre"])
    out = {}
    for i in row_indices:
        r = pf.slice(i, 1).to_pylist()[0]
        out[i] = dict(audio=read(r["audio"]["bytes"]), song_id=r["song_id"], genre=r["genre"])
    return out


def libri_utts(split):
    """[(utt_id, flac_path, text)] for a LibriSpeech split, sorted by id."""
    out = []
    for tr in sorted((LIBRI / split).glob("*/*/*.trans.txt")):
        for line in tr.read_text(encoding="utf-8").splitlines():
            uid, text = line.split(" ", 1)
            out.append((uid, tr.parent / f"{uid}.flac", text))
    return out


def durations(utts):
    return {u: sf.info(str(p)).frames / sf.info(str(p)).samplerate for u, p, _ in utts}


# ----------------------------------------------------------------------------------------------- builders
ONLY = None  # set of suite names to write (--only); None = all


class Suite:
    def __init__(self, name):
        self.name, self.man, self.chunks, self.off = name, [], [], 0

    def add(self, audio, **meta):
        audio = np.asarray(audio, dtype=np.float32)
        assert audio.ndim == 1 and len(audio) <= 30 * SR, (meta.get("id"), audio.shape)
        assert np.isfinite(audio).all()
        self.man.append(dict(offset=self.off, n=len(audio), seconds=round(len(audio) / SR, 3), **meta))
        self.chunks.append(audio)
        self.off += len(audio)

    def write(self):
        DERIVED.mkdir(parents=True, exist_ok=True)
        if ONLY is not None and self.name not in ONLY:
            print(self.name, "(not written: --only)")
            return json.loads((DERIVED / "prepare_log.json").read_text()).get(self.name)                 if (DERIVED / "prepare_log.json").exists() else None
        np.save(DERIVED / f"{self.name}.npy", np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32))
        (DERIVED / f"{self.name}.json").write_text(json.dumps(self.man, indent=0), encoding="utf-8")
        cats = {}
        for m in self.man:
            cats[m["category"]] = cats.get(m["category"], 0) + 1
        info = dict(clips=len(self.man), hours=round(self.off / SR / 3600, 3), categories=cats,
                    mb=round(self.off * 4 / 1e6, 1))
        print(self.name, info)
        return info


def silence_and_noise(s, prefix, rng, n_digital, n_near, n_noise):
    lens = lambda: int(rng.uniform(5, 30) * SR)  # noqa: E731
    for i in range(n_digital):
        n = int((5 + 25 * i / max(n_digital - 1, 1)) * SR)
        s.add(np.zeros(n, np.float32), id=f"{prefix}_sil_{i:03d}", category="silence", sub="digital_silence")
    for i in range(n_near):
        kind, db = ["white", "pink"][i % 2], [-80, -70][(i // 2) % 2]
        s.add(noise(kind, lens(), rng, db), id=f"{prefix}_nsil_{i:03d}", category="silence",
              sub=f"near_silence_{kind}_{db}dB")
    for i in range(n_noise):
        kind, db = ["white", "pink"][i % 2], [-50, -35, -20][(i // 2) % 3]
        s.add(noise(kind, lens(), rng, db), id=f"{prefix}_noise_{i:03d}", category="noise", sub=f"{kind}_{db}dB")


def environmental(s, prefix, rows, n, rng):
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["category"], []).append(r)
    classes = sorted(by_cls)
    for c in classes:
        rng.shuffle(by_cls[c])
    ptr = {c: 0 for c in classes}
    for i in range(n):
        c = classes[i % len(classes)]
        k = int(rng.integers(1, 7))
        parts, files = [], []
        for _ in range(k):
            r = by_cls[c][ptr[c] % len(by_cls[c])]
            ptr[c] += 1
            parts.append(read(ESC_DIR / "audio" / r["filename"]))
            files.append(r["filename"])
        a = np.concatenate(parts)[:30 * SR]
        s.add(a, id=f"{prefix}_env_{i:03d}", category="environmental", sub=c, files=files,
              folds=sorted(set(int(f.split("-")[0]) for f in files)), human_vocal=c in ESC_HUMAN_VOCAL)


def music(s, prefix, tracks, rng):
    for i, (idx, t) in enumerate(tracks):
        a = t["audio"]
        L = int(rng.uniform(5, min(29, len(a) / SR - 0.1)) * SR)
        st = int(rng.integers(0, max(len(a) - L, 1)))
        s.add(a[st:st + L], id=f"{prefix}_music_{i:03d}", category="music", sub=t["genre"], song_id=t["song_id"],
              start_s=round(st / SR, 2))


def pad_clip(kind, n, rng, track=None):
    if kind == "silence":
        return np.zeros(n, np.float32), "digital_silence"
    if kind == "noise":
        k, db = ["white", "pink"][int(rng.integers(2))], float(rng.uniform(-45, -35))
        return noise(k, n, rng, db), f"{k}_{db:.0f}dB"
    a = track["audio"]
    st = int(rng.integers(0, max(len(a) - n, 1)))
    seg = a[st:st + n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    db = float(rng.uniform(-35, -28))
    return set_level(seg, db), f"music_{track['song_id']}_{db:.0f}dB"


def padded(s, prefix, utts, n, rng, tracks):
    """Each utterance gets pre and post pads of 5-10 s of the same kind (silence/noise/music cycled); total <= 30 s."""
    kinds = ["silence", "noise", "music"]
    for i, (uid, path, text) in enumerate(utts[:n]):
        x = read(path)
        kind = kinds[i % 3]
        room = 30 * SR - len(x)
        pre = int(rng.uniform(5, 10) * SR)
        post = int(rng.uniform(5, 10) * SR)
        if pre + post > room:
            scale = room / (pre + post)
            pre, post = int(pre * scale), int(post * scale)
        tr = tracks[i % len(tracks)][1] if kind == "music" else None
        a, da = pad_clip(kind, pre, rng, tr)
        b, db_ = pad_clip(kind, post, rng, tr)
        s.add(np.concatenate([a, x, b]), id=f"{prefix}_{uid}", category="speech", pad=kind, pad_detail=[da, db_],
              pre_s=round(pre / SR, 2), post_s=round(post / SR, 2), speech_s=round(len(x) / SR, 2),
              utt=uid, ref=text)


def background_mix(s, prefix, utts, rng, esc, tracks):
    """Speech in background: each utterance mixed with an ESC-50 sound (one class, recordings concatenated) or a music
    excerpt over its whole length, at SNR 0 / 5 / 10 dB (RMS of the utterance vs RMS of the background segment).
    Background type alternates env / music; SNR cycles every two clips."""
    by_cls = {}
    for r in esc:
        by_cls.setdefault(r["category"], []).append(r)
    classes = sorted(by_cls)
    for i, (uid, path, text) in enumerate(utts):
        x = read(path)
        n = len(x)
        kind, snr = ("env", "music")[i % 2], BG_SNRS[(i // 2) % len(BG_SNRS)]
        for _ in range(20):
            if kind == "env":
                c = classes[int(rng.integers(len(classes)))]
                recs = [by_cls[c][j] for j in rng.permutation(len(by_cls[c]))]
                parts, files = [], []
                while sum(len(p) for p in parts) < n + SR:
                    r = recs[len(files) % len(recs)]
                    parts.append(read(ESC_DIR / "audio" / r["filename"]))
                    files.append(r["filename"])
                src, detail = np.concatenate(parts), dict(bg_class=c, bg_files=files)
            else:
                t = tracks[int(rng.integers(len(tracks)))][1]
                src = np.tile(t["audio"], int(np.ceil((n + SR) / len(t["audio"]))))
                detail = dict(bg_class=t["genre"], song_id=t["song_id"])
            st = int(rng.integers(0, max(len(src) - n, 1)))
            bg = src[st:st + n]
            if np.sqrt(np.mean(bg.astype(np.float64) ** 2)) > 1e-3:
                break
        sx = np.sqrt(np.mean(x.astype(np.float64) ** 2))
        sb = np.sqrt(np.mean(bg.astype(np.float64) ** 2))
        y = x + bg * (sx / sb / 10 ** (snr / 20))
        peak = np.abs(y).max()
        if peak > 0.99:
            y = y / peak * 0.99
        s.add(y.astype(np.float32), id=f"{prefix}_{uid}", category="speech", bg=kind, snr=snr, utt=uid, ref=text,
              **detail)


def plain_speech(s, prefix, utts):
    for uid, path, text in utts:
        s.add(read(path), id=f"{prefix}_{uid}", category="speech", utt=uid, ref=text)


def main():
    log = {}
    # ---- music tracks: shuffle the instrumental pool once, split practice / held-out
    t, keep = music_tracks()
    rng = random.Random(1234)
    rng.shuffle(keep)
    prac_idx, held_idx = keep[:50], keep[50:]
    mus = music_audio(keep)
    prac_tracks = [(i, mus[i]) for i in prac_idx]
    held_tracks = [(i, mus[i]) for i in held_idx]
    prac_tracks = [t for t in prac_tracks if t[1]["song_id"] not in EXCLUDE_SONGS]
    held_tracks = [t for t in held_tracks if t[1]["song_id"] not in EXCLUDE_SONGS]
    log["music"] = dict(source="lewtun/music_genres_small (FMA, 30 s clips)", genres=MUSIC_GENRES,
                        pool=len(keep), practice_tracks=len(prac_tracks), heldout_tracks=len(held_tracks),
                        excluded_songs=EXCLUDE_SONGS)

    esc = esc_rows()
    esc_prac = [r for r in esc if r["fold"] in ("1", "2")]
    esc_held = [r for r in esc if r["fold"] in ("3", "4", "5")]
    log["esc50"] = dict(excluded=ESC_EXCLUDE, flagged_human_vocal=ESC_HUMAN_VOCAL, classes=len({r["category"] for r in esc}),
                        practice_recordings=len(esc_prac), heldout_recordings=len(esc_held))

    # ---- held-out non-speech (300)
    s = Suite("nonspeech")
    r = np.random.default_rng(3001)
    silence_and_noise(s, "ns", r, n_digital=6, n_near=24, n_noise=60)
    environmental(s, "ns", esc_held, 140, r)
    music(s, "ns", held_tracks[:70], r)
    log["nonspeech"] = s.write()

    # ---- practice non-speech (100), other seeds / folds / tracks
    s = Suite("practice_nonspeech")
    r = np.random.default_rng(1001)
    silence_and_noise(s, "pns", r, n_digital=2, n_near=8, n_noise=20)
    environmental(s, "pns", esc_prac, 45, r)
    music(s, "pns", prac_tracks[:25], r)
    log["practice_nonspeech"] = s.write()

    # ---- LibriSpeech test-clean: padded (300) and WER (500), disjoint
    tc = libri_utts("test-clean")
    dur = durations(tc)
    rng = random.Random(42)
    order = tc[:]
    rng.shuffle(order)
    pad_pool = [u for u in order if 2.0 <= dur[u[0]] <= 18.0][:300]
    used = {u[0] for u in pad_pool}
    wer_pool = [u for u in order if u[0] not in used and dur[u[0]] <= 29.5][:500]
    s = Suite("padded")
    padded(s, "pad", pad_pool, 300, np.random.default_rng(3002), held_tracks[70:])
    log["padded"] = s.write()
    s = Suite("wer")
    plain_speech(s, "wer", wer_pool)
    log["wer"] = s.write()

    # ---- LibriSpeech dev-clean: practice speech (100) and practice padded (50), disjoint
    dc = libri_utts("dev-clean")
    dd = durations(dc)
    rng = random.Random(7)
    order = dc[:]
    rng.shuffle(order)
    sp = [u for u in order if dd[u[0]] <= 25.0][:100]
    used = {u[0] for u in sp}
    pp = [u for u in order if u[0] not in used and 2.0 <= dd[u[0]] <= 18.0][:50]
    s = Suite("practice_speech")
    plain_speech(s, "ps", sp)
    log["practice_speech"] = s.write()
    s = Suite("practice_padded")
    padded(s, "ppad", pp, 50, np.random.default_rng(1002), prac_tracks[25:])
    log["practice_padded"] = s.write()

    # ---- hard speech guards: speech in background (test-clean, disjoint from padded / wer) and test-other
    rest = [u for u in tc if u[0] not in {v[0] for v in pad_pool + wer_pool} and dur[u[0]] <= 29.5]
    random.Random(44).shuffle(rest)
    bg_pool = rest[:300]
    s = Suite("bg")
    background_mix(s, "bg", bg_pool, np.random.default_rng(3003), esc_held, held_tracks)
    log["bg"] = s.write()
    order = []
    if (LIBRI / "test-other").exists():
        to = libri_utts("test-other")
        dto = durations(to)
        order = [u for u in to if dto[u[0]] <= 29.5]
        random.Random(45).shuffle(order)
        s = Suite("other")
        plain_speech(s, "oth", order[:300])
        log["other"] = s.write()
    else:
        print("WARNING: LibriSpeech test-other not extracted; suite 'other' not built")
    used = {u[0] for u in sp + pp}
    pb = [u for u in dc if u[0] not in used and dd[u[0]] <= 25.0]
    random.Random(8).shuffle(pb)
    s = Suite("practice_bg")
    background_mix(s, "pbg", pb[:50], np.random.default_rng(1003), esc_prac, prac_tracks)
    log["practice_bg"] = s.write()

    log["speakers"] = dict(test_other=len({u[0].split("-")[0] for u in order[:300]}),
                           test_clean_bg=len({u[0].split("-")[0] for u in bg_pool}),test_clean_padded=len({u[0].split("-")[0] for u in pad_pool}),
                           test_clean_wer=len({u[0].split("-")[0] for u in wer_pool}),
                           dev_clean_practice=len({u[0].split("-")[0] for u in sp + pp}))
    (DERIVED / "prepare_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(json.dumps(log["speakers"]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated suites to (re)write; others are left untouched")
    args = ap.parse_args()
    if args.only:
        ONLY = set(args.only.split(","))
    main()
