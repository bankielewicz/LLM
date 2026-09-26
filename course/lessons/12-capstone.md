# 12 — Prove the complete workflow

**Outcome:** Deliver a small, inspectable experiment and explain its limits. **Time:** two or three sittings, with bounded training runs. **Prerequisite:** lessons 00–11.

## Your task

Choose the supplied synthetic documents or a small permitted corpus of your own. Train a model from random weights. Evaluate it, save and load it, generate from fixed prompts, and run one controlled comparison. A weak model with a careful explanation can complete this capstone; an attractive sample with no reproducible experiment cannot.

Do not set an arbitrary target loss as a graduation requirement. Loss depends on data, tokenization, and the evaluation protocol. The goal is to understand and demonstrate the process.

## Deliverables

1. A data note: source, permitted use, split method, counts, exact/near-duplicate limitations, and any deliberate subset.
2. A tokenizer check: round trips for spaces, newline, tab, accented text, and an emoji; saved tokenizer identity; vocabulary size.
3. A run folder: manifest, initial/final metrics, tokenizer, checkpoint, and result status.
4. A checkpoint demonstration: inspect, evaluate again, generate from three fixed prompts, and resume into a new folder for a small number of updates.
5. A comparison note: hypothesis, one changed setting, controls, observations, elapsed time, and limitations.
6. A short explanation of the path from input text to an optimizer update and then to generated text.

## Rubric

| Evidence | Points | Full-credit behavior |
|---|---:|---|
| Representation | 20 | Exact round trips and correct tokenizer/model distinction |
| Data | 20 | Explicit splits, recorded identity, honest leakage limits |
| Training | 20 | Fresh initialization and a traceable bounded run |
| Evaluation | 20 | Comparable validation evidence and correctly scoped claims |
| Reuse and explanation | 20 | Saved-model generation, resume, and coherent explanation |

Award partial credit for incomplete evidence and say what is missing. Self-marking a lesson as read is separate from demonstrating these outcomes. An offline explanation or simulated chart does not earn points for executed training.

## Explain these five cases

- The tokenizer changes an emoji into an unrelated letter. **Diagnosis:** representation failure; fix it before assessing model quality.
- Training improves while validation consistently worsens. **Diagnosis:** investigate generalization and data/sampling quality; do not assume more steps solve it.
- A new tokenizer has the same vocabulary size as the old one. **Diagnosis:** matching dimensions do not establish a compatible ID mapping.
- A resumed run starts at zero. **Diagnosis:** inspect whether it actually loaded the intended checkpoint; the operation may be a fresh run.
- A good story continuation contains a false factual claim. **Diagnosis:** fluent prediction does not establish factual reliability.

## Finish

Write: “This run demonstrates ____. It does not establish ____. The next experiment that would reduce my uncertainty is ____.” Link each positive claim to an actual artifact. Keep both successful and failed attempts; a failure you can diagnose is part of learning.

Then use [the glossary](../reference/GLOSSARY.md) and [sources](../reference/SOURCES.md) to choose your next topic. The core course is complete when you can explain and reproduce the workflow, not when a progress animation reaches 100%.
