# 01 — Follow one prediction

**Outcome:** Explain the complete path from text to one predicted token. **Time:** 15 minutes. **Prerequisite:** lesson 00.

## Understand

A causal language model estimates the next token given previous tokens. Its answer is a distribution over its vocabulary. Choosing a token and appending it gives the model a longer input for the next prediction.

```text
text → tokenizer → IDs → embedding vectors → transformer blocks
     → scores for all vocabulary entries → probabilities → next ID → text
```

The tokenizer determines the representation. The model learns the probabilities. A token ID is an address in a vocabulary, not a score or a meaning. For this lesson, suppose a fictional tokenizer maps `the`, ` cat`, and ` sat` to IDs 8, 19, and 42. The model does not conclude that ` sat` is more meaningful because 42 is larger than 19.

An untrained model already has the machinery to produce scores, but its random weights have not learned language patterns. Training adjusts those weights using examples. Generation uses the weights as they are. A model can produce convincing text and still be factually wrong: likely continuation and verified truth are different objectives.

## Work one example

Suppose a training sequence is `[8, 19, 42, 7]`. A three-position window gives:

```text
inputs:   [8, 19, 42]
targets:  [19, 42, 7]
```

The first prediction sees token 8 and should predict 19. The second sees 8 and 19 and should predict 42. The third sees 8, 19, and 42 and should predict 7. The causal mask prevents earlier positions from seeing the later answers even though the whole input window is processed in one training pass.

## Predict, then check

For `[4, 6, 2, 9, 3]`, write a four-token input and target. Which tokens may the second prediction use?

**Answer:** Input `[4, 6, 2, 9]`; target `[6, 2, 9, 3]`. The second prediction uses `[4, 6]` to predict 2. It cannot use the input's later 2 or 9.

**Common misconception:** Training requires a person to label every next token. For this task the text supplies the targets by shifting the sequence. Data selection and evaluation still require care.

**Stop here:** Explain the pipeline aloud without using “it just understands.” Next: [tokenizers](02-tokenizers.md).
