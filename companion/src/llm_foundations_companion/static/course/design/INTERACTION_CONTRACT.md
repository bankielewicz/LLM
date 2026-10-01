# Presentation and data contract

The new course supplies content and local Python labs. It does not supply a web application or a live-training API. Claude owns that interface design; any future backend integration needs its own implementation and verification.

## Content

`curriculum.json` orders lessons and records titles, paths, outcomes, prerequisites, and suggested sitting times. Markdown is the canonical lesson body. Relative links resolve from each Markdown file. Preserve code indentation, tables, worked examples, and answer explanations. Learner notes and completion state belong to a separate local data structure, not edits to course files.

## Tokenizer

`tokenizer.json` has `format: teaching-byte-bpe-v1` and an ordered `merges` array. IDs 0–255 map to bytes, 256 to EOS, and merge `i` creates ID `257+i`. Each merge contains two earlier non-EOS IDs. Start text encoding with UTF-8 bytes and apply merges in order. Decode by concatenating token bytes and then decoding the complete byte sequence. An individual token can contain only part of a Unicode character, so show its bytes when a standalone character display is impossible.

`examples/tokenizer-cases.json` supplies reference input/IDs/decoded values for byte and BPE modes. The tests cover actual newlines, tabs, repeated spaces, accented text, and emoji. Do not collapse whitespace in either the representation or its explanatory display.

## Recorded training

Use `examples/byte-training/metrics.jsonl` for real short-run data. Each JSON line contains:

```json
{"phase":"training","step":50,"train_loss":3.86,"validation_loss":3.86,"validation_perplexity":47.66,"elapsed_seconds":0.4}
```

The numbers above illustrate the schema; read the actual file for plotted values. The initial row has step zero. Label axes with optimizer updates and mean negative log probability in nats per token. Perplexity may be null for an unrepresentable exponent. Use the companion manifest for tokenizer, configuration, and evaluation protocol. Compare raw losses only when the data, tokenizer, and measurement protocol support that comparison.

`result.json` reports the recorded run status. Missing data is unknown, never a successful zero value. A file import supplies recorded evidence; it does not establish that a process is still running. The sample text in `sample.txt` is actual generation from the recorded byte checkpoint.

## Suggested UI states

| Area | Required distinguishable states |
|---|---|
| Lesson | Not started, reading, practiced, understanding checked |
| Simulation | Labelled simulation, paused, stepping, reset |
| Imported experiment | No data, importing, valid recorded data, incompatible/malformed data |
| Future live connection | Disconnected, connected, running, paused, interrupted, failed, completed |
| Checkpoint | Specified recorded step, best observed, latest saved, missing/incompatible |
| Progress storage | Saved locally, save unavailable, exported, import error |

These are design requirements, not evidence that a browser implementation already exists. Do not call Pause/Resume controls for recorded data “pause training.” Playback is a different operation.

## Acceptance examples

- Changing a future token leaves earlier-position predictions unavailable to that token in the causal-mask lesson.
- The tokenizer UI preserves a newline rather than turning it into a space or unrelated character.
- An imported four-step BPE run cannot be displayed as a completed 200-step byte run.
- Reading a lesson can update reading progress without awarding capstone training evidence.
- The recorded sample's repetitive continuation remains visible alongside its improved validation loss.
- A live-training view with no backend connection offers setup/command guidance rather than invented live measurements.
