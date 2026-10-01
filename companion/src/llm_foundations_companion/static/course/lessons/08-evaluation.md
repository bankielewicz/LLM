# 08 — Decide what the result actually demonstrates

**Outcome:** Interpret validation evidence without equating loss with intelligence. **Time:** 20 minutes. **Prerequisite:** one completed small training run.

## Understand

Training loss answers how well a model predicts examples from its training distribution. Validation loss estimates prediction performance on reserved documents. If training loss keeps falling while validation loss worsens consistently, investigate overfitting, split quality, sampling noise, and changed settings.

Our evaluation switches the model to evaluation mode and disables gradient tracking. These are different operations: `eval()` changes the behavior of layers such as dropout; `no_grad()` prevents gradient recording. Evaluation does not call the optimizer.

The lab measures eight fixed random batches of windows from each split. Equal seeds make repeated checkpoint comparisons use the same windows. Windows can overlap. The result is an estimate on those sampled windows, not an exhaustive benchmark or a statistical confidence interval. Compare several seeds or use broader evaluation before making stronger claims.

## Try

```text
python labs/lab.py evaluate --checkpoint runs/first/last.pt
```

**Expected:** the same checkpoint's validation result agrees with its saved evaluation, within numerical tolerance on the same environment. The command checks the original dataset hashes. A changed data file produces an error rather than an apparently comparable score.

Perplexity is `exp(average token loss)` when loss uses natural logarithms. Loss 2 means perplexity about 7.39. It is not “7.39% wrong.” Tokenization changes its units: comparing a byte model's raw perplexity with a BPE model's raw perplexity is not a fair model ranking. See [Hugging Face's perplexity discussion](https://huggingface.co/docs/transformers/perplexity).

## Work a diagnosis

Suppose measured losses are:

| Step | Training | Validation |
|---|---:|---:|
| 50 | 3.0 | 3.2 |
| 100 | 2.4 | 2.8 |
| 200 | 1.6 | 3.1 |

These values are an illustrative exercise, not results from your run. Step 100 is the best measured validation point in this table. The last checkpoint is not automatically the best one. Inspect samples, split quality, and repeated measurements before deciding the later model is broadly worse.

## Check

The model memorizes the synthetic story pattern and gets a low validation loss. Can you conclude it can answer science questions?

**Answer:** No. The validation documents share the synthetic template distribution. Evaluate the capability you intend to claim using suitable held-out tasks and data. Loss and a few attractive samples each provide limited evidence.

**Stop here:** Write one supported conclusion and one unsupported conclusion about your run. Next: [saving and generation](09-checkpoints.md).
