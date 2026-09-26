# 09 — Save the experiment, then use it

**Outcome:** Distinguish generating from saved weights, resuming training, and starting over. **Time:** 20 minutes. **Prerequisite:** lesson 08.

## Understand

A model file is not a transcript of the training text. A checkpoint stores tensors and enough configuration to rebuild the model. This lab also saves the tokenizer mapping, optimizer state, training step, random-number states, and dataset identity. These let you continue the same experiment instead of guessing how it was constructed.

| Action | Starting weights | Weight updates? |
|---|---|---|
| `train` | Fresh random values | Yes |
| `generate` | Specified checkpoint | No |
| `resume` | Specified checkpoint | Yes |

Loading a tokenizer alone does not load trained model weights. Keeping a filename while replacing its contents does not preserve experiment identity. The lab records hashes to make those differences visible.

## Try the saved model

```text
python labs/lab.py inspect --checkpoint runs/first/last.pt
python labs/lab.py generate --checkpoint runs/first/last.pt --prompt "Mira visited the " --temperature 0
python labs/lab.py generate --checkpoint runs/first/last.pt --prompt "Mira visited the " --temperature 0.8 --seed 17
```

**Expected:** the first command reports the saved step, shapes, and identities. Temperature zero selects the highest-scoring token. A positive temperature samples from scaled scores. Sampling can change wording; it cannot add knowledge that was not learned. The generated sample may be poor after a short run.

Generation stops at EOS or at the requested number of new tokens. Inputs longer than the context window are cropped to their latest tokens for each prediction. This implementation recomputes that window each time and has no KV cache.

## Continue for 100 additional updates

```text
python labs/lab.py resume --checkpoint runs/first/last.pt --out runs/resumed --steps 100
```

If the parent stopped at 200, this run targets 300. Architecture, tokenizer, batch size, learning rate, and seed come from the checkpoint. The original run stays intact. `best.pt` in the new folder is created only if this continuation beats the parent's saved best validation loss; `last.pt` is always the most recent evaluated continuation checkpoint.

CPU checks test exact uninterrupted versus resumed updates in the verified environment. Do not assume bit-for-bit agreement across GPU types, operating systems, or package versions. Move experiments with their data as well as weights; this teaching implementation records absolute data paths and refuses resume if those paths or hashes no longer match.

## Check

You stop a run with Ctrl+C between checkpoints. Which step will resume?

**Answer:** The step stored in `last.pt`. Updates since that saved evaluation boundary were not retained. Read the checkpoint instead of assuming the terminal's last update was saved.

**Stop here:** Keep a fixed prompt and compare the original and resumed checkpoint outputs. Next: [fair experiments](10-experiments.md).
