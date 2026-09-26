"""Teaching byte/BPE tokenizer. Preserves UTF-8, whitespace, and unseen characters.

This deliberately simple whole-document BPE has no production pre-tokenizer.
IDs 0..255 represent bytes; 256 is end-of-document; merges start at 257.
"""
from collections import Counter
import hashlib
import json


def merge(seq, pair, new_id):
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class Tokenizer:
    eos_id = 256

    def __init__(self, merges=()):
        self.merges = [tuple(pair) for pair in merges]
        self.pieces = {i: bytes([i]) for i in range(256)}
        for new_id, (left, right) in enumerate(self.merges, 257):
            self.pieces[new_id] = self.pieces[left] + self.pieces[right]

    @property
    def vocab_size(self):
        return 257 + len(self.merges)

    @classmethod
    def train(cls, documents, vocab_size=320):
        if not 257 <= vocab_size <= 1024:
            raise ValueError("Teaching tokenizer vocabulary must be 257..1024.")
        seqs = [list(doc.encode("utf-8")) for doc in documents]
        if not any(seqs):
            raise ValueError("Tokenizer training needs nonempty text.")
        merges = []
        while 257 + len(merges) < vocab_size:
            counts = Counter(pair for seq in seqs for pair in zip(seq, seq[1:]))
            if not counts:
                break
            # Stable tie-break: equal-frequency pairs are ordered by token IDs.
            pair, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if count < 2:
                break
            new_id = 257 + len(merges)
            merges.append(pair)
            seqs = [merge(seq, pair, new_id) for seq in seqs]
        return cls(merges)

    def encode(self, text):
        seq = list(text.encode("utf-8"))
        for new_id, pair in enumerate(self.merges, 257):
            seq = merge(seq, pair, new_id)
        return seq

    def decode(self, ids, errors="strict"):
        # EOS is metadata, not a text byte. Invalid generated UTF-8 can be
        # displayed with errors="replace"; round-trip checks use strict mode.
        return b"".join(self.pieces[i] for i in ids if i != self.eos_id).decode("utf-8", errors)

    def to_dict(self):
        return {"format": "teaching-byte-bpe-v1", "merges": self.merges}

    @classmethod
    def from_dict(cls, value):
        if value.get("format") != "teaching-byte-bpe-v1":
            raise ValueError("Unsupported tokenizer format.")
        return cls(value["merges"])

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()
