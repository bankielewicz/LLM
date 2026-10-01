# 04 — Turn addresses into learnable vectors

**Outcome:** Explain why token IDs and embedding vectors are different, and calculate a tensor shape. **Time:** 20 minutes. **Prerequisite:** lessons 01–03.

## Understand

An embedding table has one row for each vocabulary entry. An input ID selects a row. If the vocabulary has 257 entries and the model width is 64, the table has shape `[257, 64]` and contains 16,448 learned numbers.

Initially those numbers are random. The tokenizer does not bring trained embeddings into a fresh model. As model training proceeds, gradients adjust the vectors along with the other model weights. A familiar token ID has no useful learned vector until the model learns one or loads one from a trained checkpoint.

```text
IDs:                         [batch, positions]
after embedding lookup:      [batch, positions, width]
after transformer blocks:    [batch, positions, width]
after vocabulary projection: [batch, positions, vocabulary]
```

The final projection produces one score for each possible next token. Thus a vocabulary change affects the input table and output layer. In some architectures those weights are shared; our lab keeps them separate to make both operations visible.

Our model also learns a position vector for each position in the context window. Adding it to the token vector lets this architecture distinguish the order of tokens. A context window is measured in tokens; changing tokenizer changes how much text fits into that window.

## Work the shapes

Eight examples with 64 positions give input shape `[8, 64]`. With width 64, the representation becomes `[8, 64, 64]`. With the byte vocabulary, output scores have shape `[8, 64, 257]`.

Inspect `TinyLM` in `labs/model.py`: `tokens` performs the lookup, `positions` adds positional information, and `output` projects back to vocabulary scores. [PyTorch's Embedding reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) defines the lookup operation.

## Check

If the vocabulary grows from 257 to 320 while width stays 64, what changes?

**Answer:** The input table gains 63 rows; the output projection gains 63 vocabulary outputs. The intermediate width stays 64. In our separate-weight model those two matrices together gain `2 × 63 × 64 = 8,064` parameters.

**Common misconception:** A two-dimensional embedding picture proves that nearby words mean the same thing. A plotted projection hides dimensions, and similarity depends on the learned representation and context. Here, a random embedding has no established semantic interpretation.

**Stop here:** Annotate the four shapes above for batch 2, context 16, width 32, vocabulary 320. Next: [attention](05-attention.md).
