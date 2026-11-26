# Earmark cloud narrator, own-voice path. Reads chapter parts in the app owner's cloned voice with Qwen3-TTS 0.6B Base
# (Apache-2.0) on the runner's CPU, watermarks the audio (AudioSeal, MIT) and uploads headerless MP3 frames that the
# Worker joins into the chapter. On CPU this runs at roughly a fifth of real time, which is why chapters come in parts.
# Auth is GitHub's OIDC token. The voice sample is fetched from the Worker while a part is being read and lives only in
# this process's memory; it is never written to disk, the cache or the log. Logs never print book text or titles.
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

BASE = os.environ.get("EARMARK_URL", "https://listen.studiko.app").rstrip("/")
DEV = os.environ.get("EARMARK_DEV") == "1"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
OUT_SR = 24000
MP3_KBPS = 48
PIECE_CHARS = 220
SENTENCE_PAUSE = 0.2
PARAGRAPH_PAUSE = 0.55
# Stop claiming new parts early enough that a slow part still finishes inside the job's time limit.
STOP_AFTER_S = float(os.environ.get("STOP_AFTER_MIN", "200")) * 60
STARTED = time.time()


def oidc() -> str:
    if DEV:
        return "dev"
    url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    tok = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not url or not tok:
        raise RuntimeError("No OIDC token (the workflow needs permissions: id-token: write)")
    req = urllib.request.Request(f"{url}&audience=earmark-runner", headers={"authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["value"]


def call(path: str, method: str = "GET", body: bytes | None = None, ctype: str | None = None):
    headers = {"authorization": f"Bearer {oidc()}", "user-agent": "earmark-narrator"}
    if DEV:
        headers["x-dev-runner"] = "1"
    if ctype:
        headers["content-type"] = ctype
    req = urllib.request.Request(f"{BASE}{path}", data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def pieces(text: str) -> list[tuple[str, float]]:
    """Sentences grouped up to PIECE_CHARS, each with the pause that follows it."""
    out: list[tuple[str, float]] = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        sentences = re.split(r"(?<=[.!?])\s+|(?<=[.!?][\"'”’)])\s+", para)
        buf = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            while len(s) > PIECE_CHARS:
                cut = s.rfind(",", 0, PIECE_CHARS)
                cut = cut if cut > PIECE_CHARS // 2 else s.rfind(" ", 0, PIECE_CHARS)
                cut = cut if cut > 0 else PIECE_CHARS
                if buf:
                    out.append((buf, SENTENCE_PAUSE))
                    buf = ""
                out.append((s[: cut + 1].strip(), 0.12))
                s = s[cut + 1 :].strip()
            if buf and len(buf) + 1 + len(s) > PIECE_CHARS:
                out.append((buf, SENTENCE_PAUSE))
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            out.append((buf, SENTENCE_PAUSE))
        if out:
            out[-1] = (out[-1][0], PARAGRAPH_PAUSE)
    return out


class Reader:
    def __init__(self):
        import torch
        from qwen_tts import Qwen3TTSModel

        torch.set_num_threads(os.cpu_count() or 4)
        self.torch = torch
        self.model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa")
        self.prompts: dict[str, object] = {}
        try:
            from audioseal import AudioSeal

            self.marker = AudioSeal.load_generator("audioseal_wm_16bits").eval()
        except Exception:
            self.marker = None
            print("watermark model unavailable; the Worker's log still records every part")

    def prompt(self, voice_id: str):
        if voice_id not in self.prompts:
            import numpy as np
            import soundfile as sf

            status, body = call(f"/runner/clone/voices/{voice_id}")
            if status != 200:
                raise RuntimeError(f"voice fetch failed: {status}")
            v = json.loads(body)
            audio, sr = sf.read(io.BytesIO(base64.b64decode(v["wav"])), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self.prompts[voice_id] = self.model.create_voice_clone_prompt(ref_audio=(np.ascontiguousarray(audio), sr), ref_text=v["text"], x_vector_only_mode=False)
        return self.prompts[voice_id]

    def say(self, text: str, prompt):
        import numpy as np

        # A cap on new tokens (about 12 per second of speech) stops a rare run-on generation from babbling for minutes.
        # Normal speech is about 0.07 s per character; past twice that, try once more, then keep only a plausible length.
        cap = int(len(text) * 1.6) + 48
        limit = len(text) * 0.13 + 3
        for attempt in range(2):
            wavs, sr = self.model.generate_voice_clone(text=text, language="English", voice_clone_prompt=prompt, max_new_tokens=cap)
            audio = np.asarray(wavs[0], dtype="float32")
            if len(audio) / sr <= limit:
                return audio, sr
        return audio[: int(sr * limit)], sr

    def watermark(self, audio, sr: int, mark: str):
        if self.marker is None:
            return audio
        import numpy as np

        torch = self.torch
        bits = int(hashlib.sha256(mark.encode()).hexdigest()[:4], 16)
        msg = torch.tensor([[(bits >> i) & 1 for i in range(16)]])
        out = np.empty_like(audio)
        step = sr * 60
        with torch.no_grad():
            for i in range(0, len(audio), step):
                x = torch.from_numpy(np.ascontiguousarray(audio[i : i + step]))[None, None, :]
                try:
                    wm = self.marker.get_watermark(x, sample_rate=sr, message=msg)
                except TypeError:
                    wm = self.marker.get_watermark(x, message=msg)
                out[i : i + step] = (x + wm).clamp(-1, 1)[0, 0].numpy()
        return out

    def read(self, text: str, voice_id: str, mark: str):
        import numpy as np

        prompt = self.prompt(voice_id)
        chunks = []
        sr_out = OUT_SR
        for piece, pause in pieces(text):
            audio, sr = self.say(piece, prompt)
            sr_out = sr
            nz = np.flatnonzero(np.abs(audio) > 0.004)
            if len(nz):
                audio = audio[max(0, nz[0] - int(sr * 0.03)) : nz[-1] + int(sr * 0.03)]
            chunks.append(audio)
            chunks.append(np.zeros(int(sr * pause), dtype="float32"))
        chunks.append(np.zeros(int(sr_out * 0.4), dtype="float32"))
        audio = np.concatenate(chunks) if chunks else np.zeros(sr_out, dtype="float32")
        return self.watermark(audio, sr_out, mark), sr_out


def to_mp3(audio, sr: int) -> tuple[bytes, float]:
    """Constant-bitrate MP3 with no ID3 or Xing header, so the Worker can join parts by concatenation."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.wav"), os.path.join(d, "out.mp3")
        sf.write(src, audio, sr, subtype="FLOAT")
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", str(OUT_SR), "-c:a", "libmp3lame", "-b:a", f"{MP3_KBPS}k",
             "-write_xing", "0", "-id3v2_version", "0", "-write_id3v1", "0", "-map_metadata", "-1", "-f", "mp3", dst],
            check=True,
        )
        with open(dst, "rb") as f:
            mp3 = f.read()
    return mp3, len(mp3) * 8 / (MP3_KBPS * 1000)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    reader = None
    done = 0
    failed_in_a_row = 0
    while time.time() - STARTED < STOP_AFTER_S:
        status, body = call("/runner/clone/claim", "POST")
        if status != 200:
            print(f"claim failed: {status}")
            return 1
        part = json.loads(body).get("part")
        if not part:
            break
        t0 = time.time()
        try:
            reader = reader or Reader()
            audio, sr = reader.read(part["text"], part["voiceId"], part["mark"])
            mp3, duration = to_mp3(audio, sr)
            status, _ = call(f"/runner/clone/parts/{part['id']}/audio?duration={duration:.2f}", "PUT", mp3, "audio/mpeg")
            if status != 200:
                raise RuntimeError(f"upload failed: {status}")
            done += 1
            failed_in_a_row = 0
            print(f"part {part['id']} ({part['index'] + 1} of {part['count']}): {round(duration)} s of audio in {round(time.time() - t0)} s")
        except Exception as e:  # This repo is public, so its logs are too: the reason goes to the Worker, not the log.
            print(f"part {part['id']}: failed")
            try:
                call(f"/runner/clone/parts/{part['id']}/fail", "POST", json.dumps({"error": str(e)[:300]}).encode(), "application/json")
            except Exception:
                pass
            failed_in_a_row += 1
            if failed_in_a_row >= 2:
                print("two parts failed in a row; stopping this run")
                return 1
    print(f"done: {done} part(s)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("clone narrator stopped with an error")
        sys.exit(1)
