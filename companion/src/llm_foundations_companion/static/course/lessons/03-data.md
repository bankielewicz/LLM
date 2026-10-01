# 03 — Make data you can trust

**Outcome:** Prepare explicit training and validation documents with no exact overlap. **Time:** 20 minutes. **Prerequisite:** lesson 02.

## Understand

Training documents produce weight updates. Validation documents help you choose settings and checkpoints. A final test set, when available, stays outside that selection process. Repeatedly optimizing against validation results can overfit your choices to validation too.

Keep related documents together when splitting realistic data: random chunks of the same article or near-identical copies can make evaluation look easier than it is. Exact duplicate detection is useful, but it cannot detect every near duplicate or shared template.

Our input format is JSON Lines: one object per physical line, with a `text` string. Newlines inside a document are escaped by JSON and restored by the reader.

```json
{"id": "example-01", "text": "A first sentence.\nA second sentence."}
```

The lab requires two explicit paths. It does not scan a folder and accidentally train on a README, guess which file you meant, or silently substitute text when a download fails. It rejects empty inputs, exact duplicates within a split, overlap between splits, and sequences too short for the selected context.

## Try

```text
python labs/make_demo_data.py --out data/demo
```

**Expected:** 180 training documents and 36 validation documents. These are newly authored synthetic stories with shared templates and different combinations. They are intentionally easy practice data. A good validation result here means success on this small distribution, not general writing ability.

For your own dataset, export two JSONL files using a JSON serializer. Record where the text came from, its permitted use, the split method, and what you removed. Keep private or unwanted text out before training. Start with a small deliberate sample: this teaching lab rejects more than 200,000 training text bytes rather than silently truncating them.

## Follow the boundaries

For BPE, only training documents learn merges. Both splits then use that frozen tokenizer. Each encoded document receives EOS. Documents are packed into a token stream; a training window may cross EOS and attend to a previous document. This is a stated teaching simplification, not independent-document masking. Fixed-length windows avoid padding in this lab.

## Check

You split one story into overlapping windows and randomly put half in validation. Is that a strong test of unseen stories?

**Answer:** No. Many words and contexts overlap. Split at the document or related-document group level before windowing. Our supplied template data remains a weak generalization test even though its documents have no exact overlap.

**Stop here:** Record the source and limitations of your selected data. Next: [embeddings](04-embeddings.md).
