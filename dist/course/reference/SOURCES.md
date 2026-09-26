# Sources and edition maintenance

Edition date: 2026-09-26. These primary references were consulted for the redesign. The lessons are original explanations and exercises tied to the included implementation; this list is not a claim that every evolving package release has been tested.

| Reference | Used for |
|---|---|
| [PyTorch installation selector](https://pytorch.org/get-started/locally/) | Current installation choice instead of a permanently hard-coded CUDA command |
| [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial) | Gradients, optimizer updates, and evaluation mode |
| [PyTorch Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html) | Token-ID lookup and embedding dimensions |
| [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | Attention and causal masking; our code writes the operation explicitly for inspection |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer attention foundations |
| [Hugging Face: BPE](https://huggingface.co/learn/llm-course/chapter6/5) | Pair-merge learning and the byte-level representation idea |
| [Hugging Face: building a tokenizer](https://huggingface.co/learn/llm-course/en/chapter6/8) | Production tokenizer components and round-trip decoding |
| [Hugging Face: training from scratch](https://huggingface.co/docs/course/chapter7/6) | Tokenizer/config/model separation in a causal-LM training workflow |
| [Hugging Face: perplexity](https://huggingface.co/docs/transformers/perplexity) | Tokenization dependence and finite-context evaluation caveats |
| [Llama documentation](https://huggingface.co/docs/transformers/model_doc/llama) | A concrete modern decoder architecture for comparison |
| [LoRA paper](https://arxiv.org/abs/2106.09685) | Parameter-efficient adaptation as a distinct follow-on topic |

## What is stable and what must be refreshed

The distinction between tokenization and weight learning, shifted prediction targets, causal masking, and train/validation separation are core concepts. Installation commands, available wheels, hardware compatibility, model availability, recommended training libraries, and performance numbers can change.

Before publishing another edition, check the installation selector and APIs used by the labs, rerun the correctness suite and documented first-run commands, and record the tested interpreter/PyTorch/platform. Benchmark runtime on actual hardware before publishing performance promises. Keep source dates and technical verification separate from learner-comprehension and visual/usability testing.

Do not label this course “fully up to date” based only on a refreshed reading list. The delivery-specific evidence and remaining gaps are in [VALIDATION.md](../VALIDATION.md).
