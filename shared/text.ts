export function wordCount(s: string): number {
  const m = s.match(/\S+/g);
  return m ? m.length : 0;
}

/** Minutes of narration at about 155 spoken words a minute. */
export function narrationMinutes(words: number): number {
  return words / 155;
}

export function normalizeText(s: string): string {
  return s
    .replace(/\r\n?/g, "\n")
    .replace(/­/g, "") // soft hyphens
    .replace(/[​‌‍﻿]/g, "")
    .replace(/[ \t ]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export interface SpeechChunk {
  text: string;
  /** Silence to add after this chunk, in seconds (longer at paragraph ends). */
  pauseAfter: number;
}

const ABBREV = /\b(Mr|Mrs|Ms|Dr|St|Jr|Sr|Prof|Mt|vs|etc|i\.e|e\.g|No|Vol|Ch)\.$/i;

/** Splits prose into sentences, keeping abbreviations like "Mr." attached. */
export function splitSentences(paragraph: string): string[] {
  const parts = paragraph.match(/[^.!?…]+(?:[.!?…]+["'”’)\]]*|$)/g) ?? [paragraph];
  const out: string[] = [];
  for (const raw of parts) {
    const p = raw.trim();
    if (!p) continue;
    const last = out[out.length - 1];
    if (last !== undefined && (ABBREV.test(last) || /^[a-z]/.test(p))) out[out.length - 1] = `${last} ${p}`;
    else out.push(p);
  }
  return out;
}

function hardSplit(sentence: string, max: number): string[] {
  if (sentence.length <= max) return [sentence];
  const out: string[] = [];
  let rest = sentence;
  while (rest.length > max) {
    const window = rest.slice(0, max);
    let cut = Math.max(window.lastIndexOf("; "), window.lastIndexOf(", "), window.lastIndexOf(": "), window.lastIndexOf(" - "));
    if (cut < max * 0.4) cut = window.lastIndexOf(" ");
    if (cut <= 0) cut = max;
    out.push(rest.slice(0, cut + 1).trim());
    rest = rest.slice(cut + 1).trim();
  }
  if (rest) out.push(rest);
  return out;
}

/**
 * Turns chapter text into chunks short enough for Kokoro (it truncates past about 510 phonemes),
 * grouping short sentences so the voice keeps a natural flow.
 */
export function chunkForSpeech(text: string, max = 320): SpeechChunk[] {
  const chunks: SpeechChunk[] = [];
  const paragraphs = normalizeText(text).split(/\n\s*\n|\n/).map((p) => p.trim()).filter(Boolean);
  for (const para of paragraphs) {
    const pieces = splitSentences(para).flatMap((s) => hardSplit(s, max));
    let buf = "";
    for (const piece of pieces) {
      if (buf && buf.length + 1 + piece.length > max) {
        chunks.push({ text: buf, pauseAfter: 0.12 });
        buf = piece;
      } else buf = buf ? `${buf} ${piece}` : piece;
    }
    if (buf) chunks.push({ text: buf, pauseAfter: 0.45 });
  }
  return chunks;
}

/** Keeps quotes short: drops any quote over the word limit (study-guide fair use pattern). */
export function shortQuote(q: string, maxWords = 15): string | null {
  const t = q.trim().replace(/^["“]|["”]$/g, "");
  if (!t) return null;
  return wordCount(t) <= maxWords ? t : null;
}

const ROMAN_VALUES: Record<string, number> = { i: 1, v: 5, x: 10, l: 50, c: 100, d: 500, m: 1000 };

export function romanToInt(r: string): number | null {
  const s = r.toLowerCase();
  if (!/^[ivxlcdm]+$/.test(s)) return null;
  let total = 0;
  for (let i = 0; i < s.length; i++) {
    const v = ROMAN_VALUES[s[i]];
    const next = ROMAN_VALUES[s[i + 1]] ?? 0;
    total += v < next ? -v : v;
  }
  return total > 0 && total < 4000 ? total : null;
}

/** "Chapter XIV" reads as "Chapter 14": the voice model says Roman numerals letter by letter. */
export function spokenTitle(title: string): string {
  return title.replace(/\b(chapter|book|part|letter|volume|act|scene)\s+([ivxlcdm]+)\b/gi, (m, word: string, num: string) => {
    const n = romanToInt(num);
    return n ? `${word} ${n}` : m;
  });
}
