# LLM Foundations: build, measure, explain

Edition: 2026-09-26. Audience: a curious beginner who can run Python and read a small function. This is a newly authored course and executable lab package. Claude's visual interface is a separate next step; these lessons are usable as Markdown now.

Your final result is a small language model you trained from random weights, a saved tokenizer, repeatable measurements, and an explanation of what the model can and cannot do. It will be a learning model, not a general assistant.

## One path through the course

| Lesson | What you will be able to do | First sitting |
|---|---|---|
| [00 — Set up](lessons/00-setup.md) | Identify the Python environment and run the lab | 20 minutes |
| [01 — Follow one prediction](lessons/01-prediction.md) | Trace text through a model to the next token | 15 minutes |
| [02 — Tokenizers](lessons/02-tokenizers.md) | Distinguish tokenizer learning from model learning | 25 minutes |
| [03 — Prepare data](lessons/03-data.md) | Make separate training and validation inputs | 20 minutes |
| [04 — Embeddings](lessons/04-embeddings.md) | Connect token IDs, vectors, and vocabulary size | 20 minutes |
| [05 — Attention](lessons/05-attention.md) | Explain which positions may see which other positions | 25 minutes |
| [06 — Build a fresh model](lessons/06-architecture.md) | Trace the shapes through a transformer | 25 minutes |
| [07 — Train](lessons/07-training.md) | Run a bounded experiment and explain each update | 25 minutes plus measured runtime |
| [08 — Evaluate](lessons/08-evaluation.md) | Compare training and validation evidence | 20 minutes |
| [09 — Save, resume, generate](lessons/09-checkpoints.md) | Reuse exactly the model and tokenizer you trained | 20 minutes |
| [10 — Run a fair experiment](lessons/10-experiments.md) | Change one variable and explain the tradeoff | 25 minutes plus measured runtime |
| [11 — Modern models and next steps](lessons/11-modern-models.md) | Place this tiny model in the wider LLM workflow | 25 minutes |
| [12 — Capstone](lessons/12-capstone.md) | Demonstrate the full workflow with evidence | Two or three sittings |

Times are suggested reading/exercise budgets, not benchmarked training promises. Pause at each lesson's stopping point. Continue from your last completed check, not from the beginning.

## First commands

Start in this `llm-foundations-v2` folder. [Lesson 00](lessons/00-setup.md) explains environment setup for Windows and WSL. All commands below use the selected environment's `python`.

```text
python labs/make_demo_data.py --out data/demo
python labs/lab.py tokenize
python labs/lab.py train --train-data data/demo/train.jsonl --val-data data/demo/validation.jsonl --out runs/first --steps 200
python labs/lab.py evaluate --checkpoint runs/first/last.pt
python labs/lab.py generate --checkpoint runs/first/last.pt --prompt "Mira visited the "
```

Each output folder must be new. A repeated command with the same folder stops instead of replacing your work. Default runs use CPU and a byte tokenizer. GPU and learned BPE are explicit later choices.

## Materials

- [Lab workbook](labs/WORKBOOK.md): four experiments with commands, observations, and acceptance checks.
- [Glossary](reference/GLOSSARY.md): terminology used throughout the course.
- [Troubleshooting](reference/TROUBLESHOOTING.md): what to do when a command fails.
- [Sources and update policy](reference/SOURCES.md): primary references and what was checked for this edition.
- [Original-course review](REVIEW.md): strengths, defects, grade, and review limits.
- [Validation record](VALIDATION.md): what was actually executed for this delivery.
- [Claude design brief](design/CLAUDE_DESIGN_PROMPT.md): visual design and navigation handoff.

The demo data is original synthetic template text. It teaches mechanics, not broad language understanding. No model or dataset downloads occur automatically. A full training evaluation on natural text and long GPU training are outside this edition's verification.
