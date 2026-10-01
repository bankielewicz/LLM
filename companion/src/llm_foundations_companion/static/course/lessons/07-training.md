# 07 — Turn prediction mistakes into weight updates

**Outcome:** Train a small model and explain loss, gradients, and the optimizer. **Time:** 25 minutes plus actual run time. **Prerequisite:** lessons 00–06 and generated demo data.

## Understand

For each position, cross-entropy loss measures the negative log probability the model assigned to the correct next token. If that probability is 0.5, the loss is about 0.693 nats; if it is 0.1, the loss is about 2.303. The model is penalized more for being confidently wrong.

One update contains five steps:

```python
_, loss = model(inputs, shifted_targets)  # predict and measure error
optimizer.zero_grad(set_to_none=True)    # clear earlier gradients
loss.backward()                         # calculate how weights affect loss
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()                        # change weights
```

Gradients describe local sensitivity; they are not the new weights. The optimizer uses them to choose an update. Learning rate controls the update scale. A step processes a batch of windows. Our random-window sampling does not define an epoch as one exact pass through all documents.

Our model expects targets that have already been shifted. Some libraries shift labels inside their model implementation instead. Always check that contract: shifting twice trains the wrong prediction task. [PyTorch's optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial) explains the update process.

## Predict, then run

Predict whether an untrained model will initially prefer correct text strongly. Then run:

```text
python labs/lab.py train --train-data data/demo/train.jsonl --val-data data/demo/validation.jsonl --out runs/first --steps 200
```

**Expected:** an initial measurement followed by measurements at steps 50, 100, 150, and 200. Each reports training and validation loss from fixed sampled windows. Loss will often trend down on this simple data, but exact values and runtime are not promised. Do not confuse the end of 200 updates with a language-quality threshold.

`manifest.json` records the model settings, package version, tokenizer fingerprint, and input hashes. `metrics.jsonl` records actual observations. `last.pt` is the latest evaluated checkpoint. Output folders cannot overwrite earlier runs.

## Check

If you run generation ten times, have you trained the model for ten more steps?

**Answer:** No. Generation performs inference without optimizer updates. Another `train` starts fresh; `resume` continues saved training, as lesson 09 explains.

**Common snag:** The command requests CUDA but CUDA is unavailable. The lab stops with a clear error; choose CPU explicitly or fix the selected installation. Hardware fallback should not silently change your experiment.

**Stop here:** Record initial and final losses and the exact run folder. Next: [evaluation](08-evaluation.md).
