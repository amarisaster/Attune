# Provenance and implementation boundaries

Attune's sequential first-listen mode is an independent implementation built
from the project's own requirements, tests, state machine, schemas, and audio
analysis code.

## Runtime dependencies and direct foundations

- [seven-ears](https://github.com/meatwife/seven-ears), by Seven Verity and
  Sunny, MIT licensed. Attune's acoustic foundation is fetched by
  `scripts/get-seven-ears.py` and pinned to commit
  `d33e7c1929237cfbb0c77e01c55811ee9f2360e5`. The later *First Listen*
  documentation at revision `1b8998935d56b9251a0da33f703e0ac311b7aa99`
  was also studied as a behavioral reference; its implementation is not copied.
- [FFmpeg](https://ffmpeg.org/) and `ffprobe` perform bounded audio decoding,
  slicing, and format conversion. Attune invokes installed binaries and does
  not redistribute them.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) performs optional, bounded import
  of one validated YouTube or YouTube Music video. Attune does not copy or
  vendor yt-dlp source.
- [LRCLIB](https://lrclib.net/) supplies optional synchronized lyrics matched
  by title, artist, and duration. Every returned encounter preserves the lyric
  source and distinguishes verified lyrics, creator captions, automatic
  captions, and unavailable lyrics.

## Research influences

The following projects were examined to understand the design space:

- [Music for Machine Ears](https://github.com/v3nommy/Music-for-Machine-Ears)
  by **v3nommy**, studied at revision
  `6aebf1e0724f0586a5e37897d4380923d2fd25a3`. The upstream project uses the
  Music for Machine Ears License 1.0.
- [Escutário](https://github.com/SolanceLab/escutario) by **Anne Solance / SolanceLab**,
  studied at revision `6fa6e3921b6a5f0a9d61b9dbf5b9ec8889e427b7`.
  The upstream project uses the PolyForm Noncommercial License 1.0.0.
- [seven-ears First Listen](https://github.com/meatwife/seven-ears/blob/main/docs/FIRST_LISTEN.md),
  revision `1b8998935d56b9251a0da33f703e0ac311b7aa99`

No source code, prompt text, schema, constants, tests, or interface from Music
for Machine Ears or Escutário is included. The sequential requirements—bounded
passages, recording an impression before advancing, replay after restart, and
withholding whole-song context until completion—were independently implemented.

## Private integrations

The public engine uses arbitrary listener namespace strings and a
caller-supplied delivery callback for its durable journal outbox. Household
identities, credentials, CogCor endpoints, Nexus routes, OAuth configuration,
prompts, memories, and deployment paths are not part of this repository.
