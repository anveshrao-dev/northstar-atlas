# Northstar Atlas — Agent Instructions

## Project rules

- The authorized project root is the exact checkout path explicitly supplied
  by the user for the current task. Do not infer authorization from the current
  directory or a repository name. If no root has been authorized, ask first.
- Run `pwd` and verify the resolved absolute project root against that authorized
  path before working. If they do not match, stop and ask the user.
- Create or modify project files only inside that root. Do not modify sibling
  projects or initialize Git or uv in the parent workspace.
- Inspect existing files before editing. Do not delete or overwrite existing
  files without the user's approval.
- Do not follow symlinks outside the authorized root when creating or
  modifying files.
- Do not modify global Git, Python, uv, shell, or VS Code configuration.
  Do not install packages globally or use `sudo`.
- Reading existing system tools and configuration is permitted. If a command
  requires writing outside the project, explain why and ask for approval first.
- Keep the virtual environment in `.venv/`. Keep local databases, generated
  artifacts, temporary files, and project caches in `.local/`.
- Before uv commands, configure these session-local paths:

  ```bash
  export UV_CACHE_DIR="$PWD/.local/uv-cache"
  export UV_PYTHON_INSTALL_DIR="$PWD/.local/uv-python"
  export UV_PYTHON_BIN_DIR="$PWD/.local/bin"
  export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
  mkdir -p .local/tmp
  export TMPDIR="$PWD/.local/tmp"
  export PYTHONPYCACHEPREFIX="$PWD/.local/pycache"
  ```

- If Python must be downloaded, use `uv python install 3.12 --no-bin`.
  An existing Python 3.12 interpreter may be used without modifying it.
- Do not create external services, cloud resources, or access paid APIs without
  explicit approval for the current action. Historical smoke-test approvals
  do not authorize new calls.
- Do not configure a remote, create a GitHub repository, or push to GitHub
  without approval. Do not perform destructive Git operations.
- Use an existing Git identity. If it is missing, ask the user; do not invent
  author details or modify global Git configuration.
- These are working instructions, not operating-system-level filesystem
  isolation.

## Python standards

- Use Python 3.12 and manage the project and dependencies with uv.
- Keep installable Python code in `src/northstar_atlas/`.
- Prefer readable Python with type hints.
- Keep modules focused and maintainable. Avoid unnecessary abstractions.
- Explain each new dependency, its purpose, and alternatives before adding it.
- Keep `pyproject.toml` and `uv.lock` consistent. Use `uv sync --locked` when
  reproducing an existing environment.
- Validate imports through the installed package; do not mask packaging errors
  by adding `src/` to `PYTHONPATH`.

## RAG engineering standards

- Separate ingestion, retrieval, generation, and evaluation.
- Preserve document IDs, versions, and metadata.
- Make ingestion reproducible.
- Keep vector database integrations replaceable.
- Require grounded answers and verifiable citations.
- Design explicit handling for unsupported questions.
- Use synthetic documents for this project.

## Security

- Never hardcode API keys or expose secrets in logs.
- Use environment variables for credentials.
- Never commit `.env`, credentials, private keys, or generated vector databases.
- Keep `.env.example` limited to placeholders and nonsecret defaults.
- Load credentials only for authorized operations; do not request API keys in chat.
- Ignore local artifacts while keeping source documents, tests, evaluation
  datasets, `pyproject.toml`, and `uv.lock` under version control.

## Engineering communication

For every future implementation task:

1. Explain the purpose and the problem being solved.
2. Explain the design and technical decisions.
3. Explain alternatives and relevant trade-offs.
4. Implement only the approved scope.
5. Validate the implementation and show how to reproduce the checks.
6. Report errors and limitations honestly.
7. Summarize operational implications and remaining limitations.

## Scope and authorization boundaries

The repository has a local Q&A service and Streamlit MVP. Implement only the
current user-approved task, then stop for review. Future retrieval, orchestration,
evaluation and hosted-delivery work needs its own authorization.

Runtime root validation uses the operator's process-level `NORTHSTAR_PROJECT_ROOT`
or the installed source checkout; it does not authorize an agent to work in a
different directory. Compare it with the user's exact authorized root before
work. Do not use chat input, `.env` or repository naming as authority. Pinecone
target configuration and its separate owner confirmation remain required; do not
fabricate confirmation or silently change provider targets to make setup work.

Publication preparation does not authorize staging, commits, remotes, pushes,
cloud creation or deployment. Plans, backups and operational histories stay under
ignored `.local/`; preserve historical failures and source-policy hashes.
