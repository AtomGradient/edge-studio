# Edge Studio

Edge Studio is AtomGradient's local workbench for building and testing
on-device AI products on Apple Silicon. It ships as a Python package with a
local Studio UI, an API server, and CLI workflows for model preparation, local
chat, and the Neural Imprint learning demo.

Start with the developer documentation when you are building an app or
integrating the SDK:

- Developer site: <https://atomgradient.github.io/edge-developers/>

## Install

Use Python 3.11 in an isolated environment before installing Edge Studio.

With `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install edge-studio
```

With `uv`:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install edge-studio
```

Current release candidate: `0.0.1rc11`.

For a deterministic RC install:

```bash
python -m pip install edge-studio==0.0.1rc11
```

The package installs one command-line entry point:

| Command | Use |
|---|---|
| `edge` | Launch Studio, prepare models, run local chat, inspect receipts, and try the learning demo. |

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install edge-studio
edge studio
open http://127.0.0.1:18842
```

Edge Studio binds to `127.0.0.1:18842` by default. Override the host or port
with `VLM_HOST` and `VLM_PORT` when you need a different local address.

## First Model

The baseline developer-preview model is `qwen3.5-9b-4bit`.

```bash
edge models where qwen3.5-9b-4bit
edge models fetch qwen3.5-9b-4bit --source auto
```

`edge models fetch` supports ModelScope, Hugging Face, and hf-mirror sources.
The `auto` mode chooses a source order from the local network environment.

## First Chat

Run a local multi-turn chat after the model is available:

```bash
edge demo chat --model qwen3.5-9b-4bit --interactive
```

The first 9B model load can take tens of seconds. After `[chat:ready]`, ask a
few questions and the answer streams token by token. The CLI uses model-aware
generation defaults; override the output length with `--max-tokens` when needed.
Exit with `/exit`.

## Learning Demo

Run the local correction-learning demo once the baseline model is ready:

```bash
edge demo learn run \
  --sample finance_conservative_cashflow_v1 \
  --model qwen3.5-9b-4bit \
  --max-tokens 160 \
  --include-text
```

The demo uses a synthetic finance preference to exercise local receipts,
correction capture, Neural Imprint artifact generation, and a follow-up query
without bundling internal evaluation data.

For the full walkthrough, SDK integration paths, and API references, use the
Edge Developers documentation:

- <https://atomgradient.github.io/edge-developers/>

## Source Install

Use the source path when contributing to Edge Studio or testing a local checkout:

```bash
git clone https://github.com/AtomGradient/edge-studio.git
cd edge-studio
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Launch the packaged UI/API server from the source checkout:

```bash
edge studio
open http://127.0.0.1:18842
```

For frontend development, run Vite separately and keep the backend server
running:

```bash
npm --prefix frontend ci
npm --prefix frontend run dev
edge studio
```

The Vite UI runs at `http://localhost:5173`; the backend stays on
`http://127.0.0.1:18842`.

## Build a Wheel

```bash
python -m pip install --upgrade build
./scripts/build_wheel.sh
```

The build script compiles the frontend and copies `frontend/dist` into package
resources before building the wheel, so users installing the wheel do not need
Node.js.

## Release Checks

```bash
python -m pytest tests -q
npm --prefix frontend run lint
npm --prefix frontend run build
python -m build
```

## License

Edge Studio is released under the MIT License. See `LICENSE`.
