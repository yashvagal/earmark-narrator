# Earmark narrator

The cloud narrator for Earmark, a private app that reads your own DRM-free books aloud. A GitHub Actions workflow
claims queued chapters from the Earmark server, reads them with Kokoro-82M (Apache-2.0) on the runner's CPU, encodes
MP3 with LAME, and uploads the audio.

- No secrets live here. The workflow proves who it is with GitHub's OIDC token, and the server only accepts tokens
  from this repo's `narrate.yml` on `main`. Forks can't claim anything.
- Books, text and audio never touch this repo, and the logs print no titles or text.
- It runs when the app starts it (a book was queued), plus every 6 hours as a backup.

Source of truth for `shared/speech.ts` and `shared/text.ts` is the private Earmark repo; copy changes here.
