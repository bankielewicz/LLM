# Four labs, one experiment trail

Run commands from the course root in a Python environment with PyTorch. These are real local computations. The course's future browser simulations must be labelled separately. Use a fresh output folder whenever you repeat a command.

## Lab A — Representation before learning

Read lessons 00–03. Predict how a newline and an unseen emoji will behave.

```text
python labs/lab.py tokenize
python labs/lab.py tokenize --text "café 🌍"
python labs/make_demo_data.py --out data/demo
```

Record the decoded text and token counts. The byte tokenizer has 257 entries including EOS; text encoding itself adds no EOS. EOS is added at document boundaries by the data loader. Accept the lab when exact round trips hold and you can explain why no model has been trained yet.

## Lab B — Train, measure, save

Read lessons 04–08. First use a two-update mechanical check if you want to verify your environment:

```text
python labs/lab.py train --train-data data/demo/train.jsonl --val-data data/demo/validation.jsonl --out runs/check --steps 2 --eval-every 1
```

This checks the pipeline; it is not a language-quality run. Then run the first 200-update experiment from lesson 07 into `runs/first`. Record initial and final train/validation loss, parameter count, runtime, seed, and tokenizer identity. Inspect `result.json` for status. A run failure is not converted into a simulated success.

Accept the lab when a checkpoint exists, the record identifies the input data, and you can explain what each loss measures. Loss need not cross a made-up pass threshold.

## Lab C — Reuse the result

Read lesson 09. Inspect, evaluate, and generate from `runs/first/last.pt`, then resume into `runs/resumed`. Use the same three prompts on both checkpoints, the same temperature, and the same generation seed.

Accept the lab when the resumed step count is parent step plus additional steps, the original artifacts still exist, and generation loads the saved tokenizer. Report what changed in the samples without claiming that one selected sample is a benchmark.

## Lab D — Controlled change, then BPE

Read lessons 10–12. Run the two learning-rate settings and the BPE exercise. Record which measurements can be compared directly and which cannot. BPE learns merges only from training documents; validation does not train the tokenizer.

Accept the lab when your learning-rate comparison changes one factor and your tokenizer comparison discusses vocabulary size, token count, and context coverage without ranking different tokenizations by raw perplexity.

## Experiment note template

```text
Question and prediction:
Data source and split limitations:
Run folders and checkpoint steps:
Fixed settings:
Changed setting:
Tokenizer identity:
Observed losses, sample behavior, and elapsed time:
What the evidence supports:
What it does not establish:
Next experiment:
```

## Files you keep

| Artifact | Purpose |
|---|---|
| `manifest.json` | Configuration, source paths/hashes, environment, evaluation method |
| `tokenizer.json` | The exact tokenizer rules |
| `metrics.jsonl` | One measured record per evaluation point, including initial state |
| `last.pt` | Most recent fully evaluated training state |
| `best.pt` | Best measured validation checkpoint; see resumed-run rule in lesson 09 |
| `result.json` | Completion/interruption/failure record after the training loop |

Setup errors before training may produce only a terminal error and no run directory. Do not invent completion when `result.json` is absent. A successful short run proves operation, not model quality.
