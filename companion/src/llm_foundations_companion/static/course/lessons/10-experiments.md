# 10 — Change one thing and make a fair comparison

**Outcome:** Design a small experiment whose result you can interpret. **Time:** 25 minutes plus measured runtime. **Prerequisite:** lesson 09.

## Understand

A useful experiment starts with a question and a comparison rule. “Make the model better” is too broad. “At the same number of optimizer updates, does a different learning rate improve validation loss under the same tokenizer and sampling protocol?” can be measured.

Keep data, tokenizer, model dimensions, batch size, seed, and evaluation settings fixed when comparing learning rates. Use a fresh run directory for each. Report both quality and actual elapsed time; an improvement may cost more computation. One seed is an observation, not a universal rule.

## Try a controlled pair

```text
python labs/lab.py train --train-data data/demo/train.jsonl --val-data data/demo/validation.jsonl --out runs/lr-low --steps 200 --lr 0.0001
python labs/lab.py train --train-data data/demo/train.jsonl --val-data data/demo/validation.jsonl --out runs/lr-high --steps 200 --lr 0.001
```

Predict the outcome before running. Compare the measured validation trend, not only the last printed training loss. A higher learning rate could help, hurt, or be inconclusive in this short experiment.

## Then study representation

```text
python labs/lab.py train --train-data data/demo/train.jsonl --val-data data/demo/validation.jsonl --out runs/bpe --steps 200 --tokenizer bpe --vocab-size 320
python labs/lab.py tokenize --tokenizer-file runs/bpe/tokenizer.json --text "Mira visited the library."
```

Compare token counts for the same text using byte mode and this saved tokenizer. BPE may fit more characters into 64 positions, but it also grows the input/output vocabulary matrices. With the same batch size and steps, both models process the same number of token positions while seeing different amounts of underlying text.

Consequently, raw per-token loss across these tokenizers is not an apples-to-apples ranking. This lab's tokenizer experiment measures representation and demonstrates training. A stronger quality comparison needs a shared text evaluation protocol and a tokenizer-independent normalization, such as total negative log likelihood per UTF-8 byte, with careful context accounting. That extra metric is not implemented here.

## Check

Model A uses twice the width, twice the steps, and different data. Its samples look nicer. Which single change caused the improvement?

**Answer:** The comparison cannot identify one cause. Change one variable at a time or use a planned experiment that separates effects. Keep the attractive samples, but label the evidence correctly.

**Stop here:** Write your hypothesis, fixed settings, changed setting, observation, and limitation. Next: [modern model context](11-modern-models.md).
