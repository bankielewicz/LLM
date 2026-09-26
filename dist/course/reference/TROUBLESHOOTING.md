# Diagnose the first failure

| Symptom | What to inspect | Next action |
|---|---|---|
| `No module named torch` | Selected interpreter path | Install PyTorch in that environment using the official selector |
| Script not found | Working folder | Run from this course root; do not change scripts to compensate |
| CUDA requested but unavailable | Driver and PyTorch build for the selected OS | Use `--device cpu` deliberately, or fix that installation |
| GPU out of memory | Other GPU users, batch/context/model size | Stop the failed run; try a new run with a smaller batch or CPU |
| Output already exists | Chosen run folder | Choose a new folder; the lab preserves prior results |
| Not enough tokens | Token count in each split and requested context | Add suitable text or reduce context; do not duplicate validation into training |
| Identical document across splits | Dataset export and split process | Deduplicate/group related sources before splitting |
| Duplicate within one split | Repeated documents in export | Remove exact duplicate rows and record the change |
| Non-finite loss/gradient | Learning rate and input data | Retain the failed attempt; retry a smaller learning rate in a new folder |
| Width not divisible by heads | Model configuration | Choose a valid pair such as 64/4 or 96/4 |
| Resume says data changed | File paths and recorded SHA256 values | Restore the original inputs or start a clearly separate experiment |
| Resume cannot find data after moving folders | Absolute paths recorded in manifest | Keep the original data location available; portable relocation is not implemented |
| No `best.pt` in resumed folder | Whether validation improved on parent's best | Use new `last.pt` or the earlier best checkpoint explicitly |
| Checkpoint exists but generation is poor | Number of updates, corpus, validation trend | Describe observed behavior; a working pipeline need not produce good language |
| Replacement characters in generated text | Random byte sequences from a weak model | Distinguish invalid generated UTF-8 from exact input tokenizer round trips |
| Ctrl+C stopped the run | Step in saved `last.pt` | Resume from that recorded step; unsaved updates are discarded |

Do not hide errors behind synthetic charts or fallback corpora. A clear failure teaches more than an apparently successful run using different data. The script's terminal traceback is a diagnostic, not a success receipt.

The pure-Python BPE implementation is intentionally small and slow. Use the small supplied data first. For larger experiments, move to a maintained tokenizer implementation and validate its representation contract before changing the model.
