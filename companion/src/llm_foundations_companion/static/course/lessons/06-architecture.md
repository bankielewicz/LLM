# 06 — Build the model before it knows anything

**Outcome:** Identify every major part of a fresh causal transformer and explain its settings. **Time:** 25 minutes. **Prerequisite:** lesson 05.

## Understand

A model starts with a recipe, called its configuration. Vocabulary size sets how many IDs it can represent. Context sets the largest input window. Width sets the vector size at each position. Layers sets how many transformer blocks run in sequence. Heads sets how each attention calculation is divided.

```text
token embedding + position embedding
    ↓
repeat for each block:
    normalized input → attention → add original input
    normalized result → small feed-forward network → add prior result
    ↓
final normalization → vocabulary scores
```

The additions are residual connections: each block updates a running representation rather than replacing it with an unrelated one. The feed-forward network transforms each position separately, while attention exchanges information between positions.

In `labs/model.py`, `Config` defines the recipe and `TinyLM(cfg)` constructs the model with random weights. There is no pretrained model download or pretrained checkpoint in a new `train` command. Reusing tokenizer rules does not change this fact.

## Read the defaults

| Setting | Default | Interpretation |
|---|---:|---|
| context | 64 | Up to 64 previous input tokens per forward pass |
| width | 64 | 64 coordinates per position |
| layers | 2 | Two transformer blocks |
| heads | 4 | Four heads, each of width 16 |
| vocabulary | 257 for byte mode | 256 byte IDs plus EOS |

These defaults are chosen for an inspectable experiment, not because they are optimal. More parameters increase capacity and resource use, but do not guarantee a better model. If you change vocabulary or architecture after saving weights, those weights may no longer fit or may mean something different.

## Check

Which of these is valid: width 64 with heads 4, or width 64 with heads 6? Why?

**Answer:** Four heads is valid because each gets 16 coordinates. Six does not divide 64 evenly, so this implementation rejects the configuration before training.

**Boundary:** This architecture uses learned positions, LayerNorm, GELU, and ordinary multi-head attention. It omits dropout, weight tying, rotary positions, grouped-query attention, mixed precision, and distributed training. These are explicit teaching choices, not claims about all current LLMs. Lesson 11 explains the larger picture.

**Stop here:** Trace one input through the diagram and name what is learnable. Next: [training](07-training.md).
