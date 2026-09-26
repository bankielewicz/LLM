"""Explicit JSONL document inputs, split checks, and shifted windows."""
import hashlib
import json
from pathlib import Path
import torch


def read_documents(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    documents = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError(f"{path.name}:{number}: expected a nonempty text field.")
        documents.append(item["text"])
    if not documents:
        raise ValueError(f"No documents in {path}")
    if len(documents) != len(set(documents)):
        raise ValueError(f"Duplicate documents within {path.name}; deduplicate first.")
    return documents, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "documents": len(documents), "bytes": len(raw)}


def load_splits(train_path, val_path):
    train, train_info = read_documents(train_path)
    val, val_info = read_documents(val_path)
    if set(train) & set(val):
        raise ValueError("An identical document occurs in training and validation.")
    return train, val, {"train": train_info, "validation": val_info}


def encode_documents(documents, tokenizer, context):
    ids = [token for doc in documents for token in tokenizer.encode(doc) + [tokenizer.eos_id]]
    if len(ids) <= context:
        raise ValueError(f"Need at least {context + 1} tokens in each split; found {len(ids)}.")
    return torch.tensor(ids, dtype=torch.long)


def batch(data, context, size, rng, device):
    starts = torch.randint(len(data) - context, (size,), generator=rng)
    x = torch.stack([data[i:i + context] for i in starts])
    y = torch.stack([data[i + 1:i + context + 1] for i in starts])
    return x.to(device), y.to(device)
