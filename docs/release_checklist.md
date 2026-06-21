# Edge Studio Release Checklist

Use this checklist for every `edge-studio` PyPI release candidate.

Codex owns build, upload, tag, and pip-installed smoke. Claude owns review on
the committed SHA and the `edge-developers` documentation sync.

## Before Upload

- Confirm `pyproject.toml`, `edgestudio_core.__version__`, and README deterministic install examples use the same version.
- Run the package-local release tests.
- Build a fresh sdist and wheel from the committed release candidate.
- Run `twine check dist/*`.
- Ask for review on the committed SHA before uploading.

## Upload And Tag

- Upload both wheel and sdist to PyPI.
- Push an annotated Git tag matching the PEP 440 version, for example `v0.0.1rc15`.
- If the tag push fails after a successful upload, re-push the tag before announcing the release.
- Install from PyPI in a clean or pip-smoke environment with `--pre` and confirm `edge --version`.

## Post-Upload Smoke

- Run `edge export scaffold` from the PyPI-installed package.
- Unzip the exported app and run `xcodegen generate`.
- Confirm model ODR markers and the shared app scheme survive regeneration.
- Run `xcodebuild -list` and confirm there is no `KnownAssetTags` warning.

## Developer Docs Sync

The release is not complete until this section is done.

- Update `edge-developers/static/EDGE_AGENT_GUIDE.md`.
- Update `edge-developers/static/EDGE_AGENT_GUIDE.zh.md`.
- Bump the Task 1 expected `edge-studio` version.
- Bump the dependency table `edge-studio` version.
- Search `edge-developers` for stale `0.0.1rcN` references and update any release-current references.
- Keep `edge-kit`, `edge-engine`, and `edge-halo-binary` pins unchanged unless their releases changed too.
- Publish or hand off the docs update before asking external testers to retry.
