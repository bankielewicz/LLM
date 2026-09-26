# 11 — Connect the small model to current practice

**Outcome:** Separate pretraining, fine-tuning, retrieval, prompting, and inference optimizations. **Time:** 25 minutes. **Prerequisite:** lesson 10.

## What transfers

Your lab has the essential causal-LM workflow: explicit text data, a fixed representation, a randomly initialized network, shifted prediction targets, optimization, held-out measurements, saved state, and inference. Scaling requires more than adding layers: data quality, efficient computation, reproducibility, and appropriate evaluation become larger problems.

Several current decoder families use architectural choices beyond this teaching model. Rotary positional embeddings encode position through rotations in attention; RMSNorm is a different normalization; grouped-query attention lets multiple query heads share fewer key/value heads. These are examples, not requirements for every language model. Compare a named architecture's documentation rather than treating “modern transformer” as one universal recipe. [Llama's model documentation](https://huggingface.co/docs/transformers/model_doc/llama) offers a concrete reference.

A KV cache stores earlier attention keys and values during generation to avoid recomputing them. It consumes runtime memory and does not update model weights or become permanent learned memory. Lower-precision weights can reduce storage and some memory costs, but total GPU memory also includes activations, optimizer state during training, caches, and runtime overhead. A fixed claim that a particular parameter count “fits 12 GB” needs workload-specific measurement.

## Choose the operation

| Need | Candidate approach | What changes |
|---|---|---|
| Learn the mechanics of training | Tiny model from scratch | Initially random weights |
| Provide a useful instruction or example | Prompting | Current input context |
| Supply current documents | Retrieval-augmented generation | Retrieved material in context |
| Adapt an existing model's behavior | Fine-tuning, possibly LoRA | Some or all model parameters |
| Run the same model more efficiently | Inference optimization | Execution/storage choices |

LoRA trains small adapter matrices while the chosen base weights remain frozen; it still requires suitable data and evaluation. A tiny model trained on story continuations is not automatically an instruction-following chatbot. A chat interface alone does not supply instruction tuning or verify facts.

## Work a decision

You want answers grounded in a manual that changes weekly. Begin by testing retrieval plus an existing model and citations to the manual. Evaluate answer correctness and whether the cited passages support it. Fine-tuning is not a dependable substitute for refreshing changing reference material.

**Check:** Is loading a pretrained GPT-style model and training adapters equivalent to your fresh `TinyLM` run?

**Answer:** No. The starting weights and the trainable parameters differ. Both involve optimization, but the evidence and goals are different. [The LoRA paper](https://arxiv.org/abs/2106.09685) describes the adapter method.

**Stop here:** Choose one realistic next project and state why its training or retrieval approach matches its goal. Next: [capstone](12-capstone.md).
