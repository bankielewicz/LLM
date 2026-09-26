# Delivery verification

Date: 2026-09-26. Scope: newly authored lesson/lab package and targeted original-course checks. These checks were performed by the authoring agent; this is not an independent educational audit.

## Executed

- Seven behavior tests passed in 10.768 seconds. They cover byte/BPE Unicode and whitespace round trips, tokenizer serialization identity, shifted targets including a single valid window, insufficient data rejection, cross-split exact duplicate rejection, future-token masking, evaluation without changing weights, and an integrated train/save/resume/generate/evaluate path.
- The integration test compared six uninterrupted CPU updates against four updates plus two resumed updates. All final model tensors matched exactly. It also verified refusal to reuse an existing output folder and refusal to resume after training data changed.
- The documented synthetic-data generator created 180 training and 36 validation documents.
- The default byte-model exercise completed 200 CPU updates. Validation loss moved from **5.565559** to **2.214871** on the lab's fixed sampled windows. Re-evaluating the saved checkpoint reproduced the final value.
- A four-update BPE run completed using 320 vocabulary entries. Its validation loss was 5.582946 after four updates; this is a mechanical check, not a comparison with the byte run or a quality qualification.
- Saved-model generation executed. Greedy output repeated “the,” despite the improved loss. This is retained as evidence of limited model quality.
- Two targeted original-tokenizer checks reproduced newline corruption and seven-token README auto-selection. No full original-course training campaign was executed.

Environment: WSL Ubuntu, Python/PyTorch from the existing tutorial environment; PyTorch **2.11.0+cu128**, tests and runs explicitly on CPU with two PyTorch threads. CUDA availability was observed, but GPU training was not tested. No dependencies were installed or upgraded.

Timing inside `metrics.jsonl` starts after data encoding and model setup; it is not total wall-clock time. Do not reuse it as an installation, tokenizer-training, or universal hardware performance estimate.

## Not performed

- Rendered visual, responsive, keyboard, contrast, or screen-reader assessment. The browser tool's administrator policy rejected the localhost preview; no alternate route was used.
- Claude's visual redesign or backend integration.
- Native Windows execution, a GPU training run, large-corpus training, or production-scale tokenizer performance testing.
- Independent learner testing, comprehensive natural-language capability evaluation, or a held-out final test campaign.
- A claim of general chatbot competence or optimal hyperparameters.

## Source preservation and follow-up

The original `llm-tutorials` is preserved beside this new directory. `original-source-manifest.json` records 88 original non-environment files for a final unchanged-content check. The package's final structural/source-preservation results are written to `verification-summary.json`.

The live workspace includes `data/demo` and the executed `runs/first` and `runs/bpe-check` examples. Do not rerun their creation commands with the same output names: inspect them or use a fresh name. The handoff archive excludes those working data/checkpoint folders and includes compact recorded examples instead, so its fresh-start commands remain usable after extraction.

Longer training, additional language data, and visual design should keep their own evidence and limitations. A passed mechanical check is one condition for a useful course, not proof that the curriculum has been validated with learners.
