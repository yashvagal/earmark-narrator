// How a chapter or an episode script becomes spoken segments. Shared by the browser recorder and the cloud
// narrator (runner/narrate.ts) so both read a book the same way.
import { chunkForSpeech, spokenTitle } from "./text";
import type { EpisodeContent } from "./types";

export interface Segment {
  text: string;
  voice: string;
  pauseAfter: number;
}

export function narrationSegments(title: string, text: string, voice: string): Segment[] {
  const head: Segment = { text: spokenTitle(title.replace(/\s+/g, " ").trim()).replace(/([^.!?])$/, "$1."), voice, pauseAfter: 0.9 };
  const squash = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "");
  const chunks = chunkForSpeech(text);
  // Most books repeat the chapter heading as the first line; don't read it twice.
  if (chunks.length > 1 && (squash(chunks[0].text) === squash(title) || /^(chapter|book|part|letter)\s+[ivxlcdm\d]+\.?$/i.test(chunks[0].text.trim()))) chunks.shift();
  return [head, ...chunks.map((c) => ({ text: c.text, voice, pauseAfter: c.pauseAfter }))];
}

export function episodeSegments(ep: EpisodeContent, voiceA: string, voiceB: string): Segment[] {
  const out: Segment[] = [];
  for (const line of ep.lines) {
    const voice = line.speaker === "B" ? voiceB : voiceA;
    const chunks = chunkForSpeech(line.text);
    chunks.forEach((c, i) => out.push({ text: c.text, voice, pauseAfter: i === chunks.length - 1 ? 0.35 : 0.12 }));
  }
  return out;
}
