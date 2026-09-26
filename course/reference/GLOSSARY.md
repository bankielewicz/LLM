# Course glossary

| Term | Meaning in this course |
|---|---|
| Token | One vocabulary entry representing a byte, merged byte sequence, or special marker |
| Token ID | The integer address of an entry; its magnitude does not encode meaning |
| Vocabulary | The complete mapping between IDs and pieces, including special entries |
| BPE | A tokenizer algorithm that learns frequent adjacent-pair merges |
| EOS | End-of-document marker; ID 256 in this lab |
| Embedding | A learned vector looked up using a token ID |
| Width | Number of coordinates in each position's representation |
| Context window | Maximum number of input token positions used in a forward pass |
| Attention | A learned, weighted mixture of information from allowed positions |
| Causal mask | A restriction preventing a position from reading future positions |
| Logit | An unnormalized score before conversion to a probability distribution |
| Softmax | A conversion from scores to positive probabilities that sum to one |
| Cross-entropy loss | Here, the average negative log probability of correct next tokens |
| Gradient | Local sensitivity of a quantity such as loss to a parameter |
| Optimizer | The algorithm that uses gradients and its state to update parameters |
| Learning rate | A control on the size of optimizer updates |
| Batch | Several training windows processed together |
| Step | One optimizer update in this lab |
| Epoch | Typically one pass over a dataset; not the unit used by our random-window sampler |
| Validation | Reserved examples used to monitor and select a model or settings |
| Test set | A further held-out set used after model and setting choices are made |
| Overfitting | Improvement on training examples that fails to transfer as intended |
| Perplexity | Exponentiated mean token loss, dependent on tokenization and evaluation setup |
| Checkpoint | Saved model state, with the configuration and supporting state needed for reuse |
| Inference | Using model weights to compute predictions without optimizer updates |
| Temperature | A sampling control that rescales scores; zero means greedy in this lab |
| Pretraining | Initial model training on a broad prediction objective, often from random weights |
| Fine-tuning | Further training beginning with already trained model weights |
| LoRA | Training small adapter matrices for selected layers while freezing chosen base weights |
| RAG | Retrieving external information and supplying it in the model's input context |
| KV cache | Stored attention keys/values used to reduce recomputation during generation |
| Quantization | Representing quantities with a lower-precision scheme; total resource use still needs measurement |
| Seed | An initialization for a pseudorandom process; not a cross-platform reproducibility guarantee |
| Hash/fingerprint | A compact identity check for exact contents; not proof of data quality |

Definitions are scoped to the teaching lab. For architecture- and library-specific behavior, consult [the primary references](SOURCES.md).
