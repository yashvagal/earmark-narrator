// Copied from yashvagal/book-audio shared/ (the narrator only needs the episode types).
export type EpisodeLength = "brief" | "episode" | "deep";
export interface EpisodeLine {
  speaker: "A" | "B";
  text: string;
}

export interface EpisodeContent {
  title: string;
  hosts: number;
  length: EpisodeLength;
  lines: EpisodeLine[];
}
