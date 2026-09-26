# 05 — Let each position use its available context

**Outcome:** Explain queries, keys, values, and the causal mask without treating them as human thought. **Time:** 25 minutes. **Prerequisite:** lesson 04.

## Understand

Attention mixes information between token positions. Each position's vector is projected into a query, key, and value. A query is compared with keys to produce scores. Scores become nonnegative weights that sum to one across the allowed positions. Those weights combine the value vectors.

For one head, the calculation is:

```text
scores = query × key_transpose / sqrt(head_width)
weights = softmax(mask_future_positions(scores))
result = weights × value
```

The projections are learned numerical transformations. “Query” and “key” describe their roles; they are not literal questions and database keys. Multiple heads let the model form several mixtures, which are combined into one output vector per position.

## Work the mask

For a four-position training window, allowed attention has this pattern:

```text
           can read position
           0  1  2  3
query 0    ✓  ×  ×  ×
query 1    ✓  ✓  ×  ×
query 2    ✓  ✓  ✓  ×
query 3    ✓  ✓  ✓  ✓
```

Position 1 is trying to predict the token at position 2. Letting it read position 2 during training would reveal the answer. In the code, forbidden scores become negative infinity before softmax, giving them zero weight.

Open `Attention.forward` in `labs/model.py`. Follow the reshape from `[batch, positions, width]` to `[batch, heads, positions, head_width]`, then back again. Width must divide evenly by the number of heads. The [original Transformer paper](https://arxiv.org/abs/1706.03762) introduced the attention architecture underlying this explanation; our small decoder is a teaching adaptation.

## Predict, then check

If you replace the last token of a four-token input, should scores at the first two positions change?

**Answer:** No, for this causal model in evaluation mode. Those positions cannot attend to the changed future token. The regression checks include this property; a colorful heatmap alone would not establish it.

**Common misconception:** A high attention weight is a complete explanation of why a model answered something. Many layers and other transformations also affect the result. Treat an attention view as a view into one operation.

**Stop here:** Draw the allowed mask for three positions. Next: [the whole model](06-architecture.md).
