"""Bounded, local training lab. Run `python labs/lab.py --help`.

Every new training/resume run requires an unused output directory.
No implicit downloads, fallback data, or silent device changes.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import time
import torch

from data import load_splits, encode_documents, batch
from model import Config, TinyLM
from tokenizer import Tokenizer


def device_for(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Check your PyTorch installation, or choose --device cpu.")
    return torch.device(name)


def read_checkpoint(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("format") != "foundations-lab-v1":
        raise ValueError("Unsupported checkpoint format.")
    tokenizer = Tokenizer.from_dict(blob["tokenizer"])
    if tokenizer.fingerprint() != blob["tokenizer_sha256"]:
        raise ValueError("Tokenizer fingerprint mismatch.")
    return blob, tokenizer


def load_model(blob, device):
    model = TinyLM(Config(**blob["config"])).to(device)
    model.load_state_dict(blob["model"])
    return model


@torch.no_grad()
def evaluate(model, data, batch_size, seed, device, batches=8):
    was_training = model.training
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        x, y = batch(data, model.cfg.context, batch_size, rng, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train(was_training)
    loss = sum(losses) / len(losses)
    if not math.isfinite(loss):
        raise ValueError("Evaluation loss is non-finite.")
    return loss


@torch.no_grad()
def generate(model, tokenizer, prompt, count, temperature, seed, device):
    if not prompt:
        raise ValueError("Provide a nonempty prompt.")
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    for _ in range(count):
        logits, _ = model(ids[:, -model.cfg.context:])
        logits = logits[:, -1, :]
        if temperature == 0:
            next_id = logits.argmax(-1, keepdim=True)
        else:
            next_id = torch.multinomial((logits / temperature).softmax(-1), 1, generator=rng)
        if next_id.item() == tokenizer.eos_id:
            break
        ids = torch.cat((ids, next_id), dim=1)
    return tokenizer.decode(ids[0].tolist(), errors="replace")


def train(args):
    device = device_for(args.device)
    if args.steps < 1 or args.eval_every < 1 or args.batch < 1 or args.lr <= 0:
        raise ValueError("Steps, evaluation interval, batch size, and learning rate must be positive.")
    if args.out.exists():
        raise ValueError(f"Output already exists: {args.out}. Choose a new run directory.")
    torch.manual_seed(args.seed)
    prior = None
    if args.command == "resume":
        prior, tok = read_checkpoint(args.checkpoint)
        paths = prior["data"]
        docs, val_docs, provenance = load_splits(paths["train"]["path"], paths["validation"]["path"])
        for split in paths:
            if paths[split]["sha256"] != provenance[split]["sha256"]:
                raise ValueError(f"{split} data changed since checkpoint. Resume refused.")
        cfg = Config(**prior["config"])
        args.batch, args.lr, args.seed = prior["batch_size"], prior["learning_rate"], prior["seed"]
        model = load_model(prior, device)
    else:
        docs, val_docs, provenance = load_splits(args.train_data, args.val_data)
        if sum(len(doc.encode("utf-8")) for doc in docs) > 200_000:
            raise ValueError("This teaching lab accepts at most 200,000 training text bytes. Select a documented subset explicitly.")
        tok = Tokenizer() if args.tokenizer == "byte" else Tokenizer.train(docs, args.vocab_size)
        cfg = Config(tok.vocab_size, args.context, args.width, args.heads, args.layers)
        model = TinyLM(cfg).to(device)
    train_ids = encode_documents(docs, tok, cfg.context)
    val_ids = encode_documents(val_docs, tok, cfg.context)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = torch.Generator().manual_seed(args.seed)
    step, best = 0, float("inf")
    if prior:
        opt.load_state_dict(prior["optimizer"])
        rng.set_state(prior["batch_rng"])
        torch.set_rng_state(prior["torch_rng"])
        if device.type == "cuda" and prior.get("cuda_rng"):
            torch.cuda.set_rng_state_all(prior["cuda_rng"])
        step, best = prior["step"], prior["best_validation_loss"]
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = {"format": "foundations-lab-v1", "config": asdict(cfg), "tokenizer": tok.to_dict(),
                "tokenizer_sha256": tok.fingerprint(), "data": provenance, "seed": args.seed,
                "batch_size": args.batch, "learning_rate": args.lr, "device": str(device),
                "python": platform.python_version(), "torch": str(torch.__version__),
                "train_tokens": len(train_ids), "validation_tokens": len(val_ids),
                "parameters": sum(p.numel() for p in model.parameters()),
                "parent_checkpoint": str(args.checkpoint.resolve()) if prior else None,
                "evaluation": {"method": "8 fixed random windows per batch member", "batches": 8,
                               "seed": args.seed + 1000, "context": cfg.context}}
    (args.out / "tokenizer.json").write_text(json.dumps(tok.to_dict(), indent=2), encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def checkpoint(name):
        blob = {**metadata, "model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                "best_validation_loss": best, "batch_rng": rng.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        temp = args.out / (name + ".tmp")
        torch.save(blob, temp)
        temp.replace(args.out / name)

    def metric(phase):
        nonlocal best
        train_loss = evaluate(model, train_ids, args.batch, args.seed + 1000, device)
        val_loss = evaluate(model, val_ids, args.batch, args.seed + 1000, device)
        row = {"phase": phase, "step": step, "train_loss": train_loss, "validation_loss": val_loss,
               "validation_perplexity": math.exp(val_loss) if val_loss < 700 else None,
               "elapsed_seconds": round(time.monotonic() - started, 3)}
        with (args.out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if val_loss < best:
            best = val_loss
            checkpoint("best.pt")
        checkpoint("last.pt")

    metric("initial")
    model.train()
    final_step = step + args.steps
    status = "completed"
    try:
        while step < final_step:
            x, y = batch(train_ids, cfg.context, args.batch, rng, device)
            _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss. Reduce learning rate and inspect the data.")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            step += 1
            if step % args.eval_every == 0 or step == final_step:
                metric("training")
    except KeyboardInterrupt:
        status = "interrupted"
        # last.pt remains the most recent fully saved evaluation boundary.
        print("Interrupted. Resume from last.pt; updates since its saved step were not retained.")
    except Exception as error:
        (args.out / "result.json").write_text(json.dumps({"status": "failed", "step": step, "error": str(error)}), encoding="utf-8")
        raise
    (args.out / "result.json").write_text(json.dumps({"status": status, "step": step, "requested_final_step": final_step,
                                                     "best_validation_loss": best}, indent=2), encoding="utf-8")
    print(f"{status}: {args.out}. best.pt in a resumed run exists only if validation improved on the parent best.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "resume"):
        p = subs.add_parser(command)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--steps", type=int, default=200, help="additional optimizer updates")
        p.add_argument("--eval-every", type=int, default=50)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--lr", type=float, default=0.0003)
        p.add_argument("--seed", type=int, default=17)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        if command == "resume":
            p.add_argument("--checkpoint", type=Path, required=True)
        else:
            p.add_argument("--train-data", type=Path, required=True)
            p.add_argument("--val-data", type=Path, required=True)
            p.add_argument("--tokenizer", choices=("byte", "bpe"), default="byte")
            p.add_argument("--vocab-size", type=int, default=320)
            p.add_argument("--context", type=int, default=64)
            p.add_argument("--width", type=int, default=64)
            p.add_argument("--heads", type=int, default=4)
            p.add_argument("--layers", type=int, default=2)
    p = subs.add_parser("generate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", default="Mira visited the ")
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p = subs.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p = subs.add_parser("tokenize")
    p.add_argument("--tokenizer-file", type=Path)
    p.add_argument("--text", default="Hello\nworld 🌍")
    p = subs.add_parser("inspect")
    p.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command in ("train", "resume"):
        train(args)
    elif args.command == "tokenize":
        tok = Tokenizer.from_dict(json.loads(args.tokenizer_file.read_text(encoding="utf-8"))) if args.tokenizer_file else Tokenizer()
        ids = tok.encode(args.text)
        print(json.dumps({"text": args.text, "ids": ids, "decoded": tok.decode(ids), "round_trip": tok.decode(ids) == args.text,
                          "vocab_size": tok.vocab_size, "token_count": len(ids)}, ensure_ascii=False, indent=2))
    else:
        blob, tok = read_checkpoint(args.checkpoint)
        if args.command == "inspect":
            print(json.dumps({key: blob[key] for key in ("config", "step", "parameters", "tokenizer_sha256", "data")}, indent=2))
            return
        device = device_for(args.device)
        model = load_model(blob, device)
        if args.command == "generate":
            if args.tokens < 1 or args.temperature < 0:
                raise ValueError("Token count must be positive; temperature must be nonnegative.")
            print(generate(model, tok, args.prompt, args.tokens, args.temperature, args.seed, device))
        else:
            _, docs, info = load_splits(blob["data"]["train"]["path"], blob["data"]["validation"]["path"])
            if any(info[s]["sha256"] != blob["data"][s]["sha256"] for s in info):
                raise ValueError("Data changed since checkpoint; evaluation refused.")
            ids = encode_documents(docs, tok, model.cfg.context)
            loss = evaluate(model, ids, blob["batch_size"], blob["seed"] + 1000, device)
            print(json.dumps({"validation_loss": loss, "validation_perplexity": math.exp(loss) if loss < 700 else None,
                              "method": blob["evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
