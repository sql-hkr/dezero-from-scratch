# DeZero from Scratch

[![Image from Gyazo](https://i.gyazo.com/5d206a80f1ae158f2a2f9f5c56f1010e.png)](https://gyazo.com/5d206a80f1ae158f2a2f9f5c56f1010e)

This repository is a learning record of rebuilding [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3) completely from scratch.
Compared to the original implementation, the following differences are introduced:
- Uses [uv](https://docs.astral.sh/uv/), a fast Python package manager
- Adds [type hints](https://docs.python.org/ja/3/library/typing.html) for better readability and tooling support
- Resolves type errors with [Pyright](https://github.com/microsoft/pyright) checking
- No GPU support, to keep the code simple and easy to follow

## Setup

```bash
git clone https://github.com/sql-hkr/dezero-from-scratch.git
cd dezero-from-scratch
uv venv
source .venv/bin/activate
uv sync
```
