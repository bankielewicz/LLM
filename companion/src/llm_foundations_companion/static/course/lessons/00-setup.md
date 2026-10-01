# 00 — Know what you are running

**Outcome:** Run a command in the intended Python environment and explain where its output will go. **Time:** one 20-minute sitting. **Prerequisite:** basic terminal use.

## Understand

The course reader, Python program, and trained model are three different things. A browser displays a lesson. Python executes the lab. A checkpoint stores the numbers learned by the model. Opening a lesson does not start training.

Choose Windows Python or WSL Python for an entire experiment. An environment created in one cannot simply be activated in the other. Your working folder controls relative paths such as `data/demo`; your Python environment controls which packages are available.

From the course root, create a new environment if you need one:

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip --version
```

```bash
# WSL / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip --version
```

For Windows you can use the explicit `.\.venv\Scripts\python.exe` path instead of changing execution policy to activate it. Substitute it for `python` in course commands.

Use the current [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for your OS and CPU/GPU choice. Run the selected command with the same environment's Python. This lab needs `torch`; it does not require a model hub, API key, Transformers, or a separate tokenizer package. A GPU name alone does not establish software compatibility or available memory.

## Try

```text
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
python labs/lab.py --help
python labs/lab.py tokenize
```

**Expected:** an interpreter path, a PyTorch version, a Boolean for CUDA availability, command help, and a tokenization report with `round_trip: true`. CUDA being false is acceptable for the CPU path. The final command performs no model training.

## Check your understanding

You installed PyTorch successfully in Windows, but WSL says `No module named torch`. Is the model broken?

**Answer:** No model has run yet. WSL is using a different Python environment. Install into or select the intended environment; record `sys.executable` before trying again.

**Common snag:** a command cannot find `labs/lab.py`. Return to the `llm-foundations-v2` root. Avoid guessing a different script with the same filename.

**Stop here:** Save the interpreter path and PyTorch version in your notes. Next: [follow one prediction](01-prediction.md).
