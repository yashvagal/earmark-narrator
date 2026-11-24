// Earmark cloud narrator. GitHub Actions runs this on demand (and on a schedule as a backup); it claims queued
// chapters from the Worker, reads them with Kokoro-82M (Apache-2.0) on the runner's CPU, encodes MP3 with LAME (LGPL, unmodified) and
// uploads it. Auth is GitHub's OIDC token, so the workflow holds no secrets. Logs never print book text or titles.
import { Mp3Encoder } from "@breezystack/lamejs";
import { KokoroTTS } from "kokoro-js";
import { episodeSegments, narrationSegments, type Segment } from "./shared/speech";
import type { EpisodeContent } from "./shared/types";

const BASE = (process.env.EARMARK_URL || "https://listen.studiko.app").replace(/\/$/, "");
const DEV = process.env.EARMARK_DEV === "1";
const SAMPLE_RATE = 24000;
const MP3_KBPS = 48;
const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
// Full precision: on the runners' CPUs it is about twice as fast as q8 and keeps the voice clean.
const DTYPE = "fp32";
// Long chapters upload their opening first so playback can start while the rest is read.
const OPENING_SEC = 75;
const OPENING_MIN_CHARS = 4000;
// Stop claiming new chapters before the job's time limit so a chapter is never cut off half way.
const STOP_AFTER_MS = Number(process.env.STOP_AFTER_MIN || "300") * 60_000;
const started = Date.now();

async function oidc(): Promise<string> {
  if (DEV) return "dev";
  const url = process.env.ACTIONS_ID_TOKEN_REQUEST_URL;
  const tok = process.env.ACTIONS_ID_TOKEN_REQUEST_TOKEN;
  if (!url || !tok) throw new Error("No OIDC token (the workflow needs permissions: id-token: write)");
  const res = await fetch(`${url}&audience=earmark-runner`, { headers: { authorization: `Bearer ${tok}` } });
  if (!res.ok) throw new Error(`OIDC request failed: ${res.status}`);
  return ((await res.json()) as { value: string }).value;
}

async function call(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  headers.set("authorization", `Bearer ${await oidc()}`);
  if (DEV) headers.set("x-dev-runner", "1");
  return fetch(`${BASE}${path}`, { ...init, headers });
}

interface Job {
  id: string;
  kind: "narration" | "episode";
  title: string;
  text?: string;
  episode?: EpisodeContent;
  voice: string;
  voiceB: string;
}

function toInt16(f: Float32Array): Int16Array {
  const out = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++) {
    const s = Math.max(-1, Math.min(1, f[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function trimSilence(a: Float32Array): Float32Array {
  const th = 0.004;
  let s = 0;
  let e = a.length - 1;
  while (s < e && Math.abs(a[s]) < th) s++;
  while (e > s && Math.abs(a[e]) < th) e--;
  const pad = Math.floor(SAMPLE_RATE * 0.03);
  return a.subarray(Math.max(0, s - pad), Math.min(a.length, e + pad));
}

function join(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

async function render(
  tts: KokoroTTS,
  segments: Segment[],
  onOpening?: (mp3: Uint8Array, duration: number) => Promise<void>,
): Promise<{ mp3: Uint8Array; duration: number }> {
  const encoder = new Mp3Encoder(1, SAMPLE_RATE, MP3_KBPS);
  const parts: Uint8Array[] = [];
  let samples = 0;
  const encode = (pcm: Int16Array) => {
    const block = 1152 * 16;
    for (let i = 0; i < pcm.length; i += block) {
      const mp3 = encoder.encodeBuffer(pcm.subarray(i, i + block));
      if (mp3.length) parts.push(new Uint8Array(mp3));
    }
    samples += pcm.length;
  };
  encode(new Int16Array(Math.floor(SAMPLE_RATE * 0.25)));
  let openingSent = !onOpening;
  for (const seg of segments) {
    const out = await tts.generate(seg.text, { voice: seg.voice as never });
    encode(toInt16(trimSilence(out.audio as Float32Array)));
    encode(new Int16Array(Math.floor(SAMPLE_RATE * seg.pauseAfter)));
    if (!openingSent && samples / SAMPLE_RATE >= OPENING_SEC) {
      openingSent = true;
      // Frames so far from the same encoder, so the full file later starts with exactly these bytes.
      // Constant bitrate: the playable length is the byte count at 48 kbps.
      const opening = join(parts);
      await onOpening!(opening, (opening.length * 8) / (MP3_KBPS * 1000)).catch(() => {});
    }
  }
  encode(new Int16Array(Math.floor(SAMPLE_RATE * 0.6)));
  const tail = encoder.flush();
  if (tail.length) parts.push(new Uint8Array(tail));
  return { mp3: join(parts), duration: samples / SAMPLE_RATE };
}

async function main() {
  let tts: KokoroTTS | null = null;
  let done = 0;
  let failedInARow = 0;
  while (Date.now() - started < STOP_AFTER_MS) {
    const res = await call("/runner/claim", { method: "POST" });
    if (!res.ok) throw new Error(`claim failed: ${res.status}`);
    const { job } = (await res.json()) as { job: Job | null };
    if (!job) break;
    const t0 = Date.now();
    try {
      tts ??= await KokoroTTS.from_pretrained(MODEL_ID, { dtype: DTYPE, device: "cpu" });
      const segments = job.kind === "episode" ? episodeSegments(job.episode!, job.voice, job.voiceB) : narrationSegments(job.title, job.text ?? "", job.voice);
      const long = job.kind === "narration" && (job.text?.length ?? 0) >= OPENING_MIN_CHARS;
      const sendOpening = async (opening: Uint8Array, secs: number) => {
        const r = await call(`/runner/jobs/${job.id}/audio?partial=1&duration=${secs.toFixed(2)}`, { method: "PUT", body: opening, headers: { "content-type": "audio/mpeg" } });
        if (r.ok) console.log(`job ${job.id}: opening ready after ${Math.round((Date.now() - t0) / 1000)} s`);
      };
      const { mp3, duration } = await render(tts, segments, long ? sendOpening : undefined);
      const up = await call(`/runner/jobs/${job.id}/audio?duration=${duration.toFixed(2)}`, { method: "PUT", body: mp3, headers: { "content-type": "audio/mpeg" } });
      if (!up.ok) throw new Error(`upload failed: ${up.status} ${(await up.text()).slice(0, 200)}`);
      done++;
      failedInARow = 0;
      console.log(`job ${job.id}: ${job.kind}, ${Math.round(duration)} s of audio in ${Math.round((Date.now() - t0) / 1000)} s`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      // This repo is public, so its logs are too: the reason goes to the Worker, not the log.
      console.log(`job ${job.id}: failed`);
      await call(`/runner/jobs/${job.id}/fail`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ error: msg }) }).catch(() => {});
      // Two failures in a row usually means the runner itself is broken (no model download, no network):
      // stop instead of burning every job's three tries.
      if (++failedInARow >= 2) throw new Error("two chapters failed in a row; stopping this run");
    }
  }
  console.log(`done: ${done} chapter(s)`);
}

main().catch((e) => {
  console.error(e instanceof Error && /^(claim|two chapters)/.test(e.message) ? e.message : "narrator stopped with an error");
  process.exit(1);
});
