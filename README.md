# Edge Studio

Edge Studio is a local Apple Silicon workbench for on-device AI development:
model readiness checks, local demos, Neural Imprint artifact workflows, and
device-agent export.

This repository is the public entry point for Edge Studio. It hosts product
links, release notes, issue tracking, and support material. The installable
Python package is distributed on PyPI as `edge-studio`.

## Install

Developer Preview releases are published as pre-release Python packages:

```bash
python -m pip install --upgrade --pre edge-studio
edge doctor
```

After the first stable release:

```bash
python -m pip install edge-studio
edge doctor
```

## Documentation

- Developer docs: <https://atomgradient.github.io/edge-developers/>
- PyPI package: <https://pypi.org/project/edge-studio/>

## Repository Scope

This public repository is intentionally lightweight. It is not the source
checkout used for internal development or release automation.

Use this repository for:

- Issues and support requests
- Release notes
- Product and documentation links
- License and security contact information

## Privacy Boundary

Edge Studio workflows are designed for local, user-controlled development.
Model files, local receipts, and Neural Imprint artifacts stay on the user's
machine unless the user explicitly moves them.
