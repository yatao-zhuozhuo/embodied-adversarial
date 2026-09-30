#!/usr/bin/env bash
set -euo pipefail

# Install only the TileLang backend and its missing runtime dependencies into
# an isolated shared directory.  Do not let the resolver replace Torch or its
# bundled Triton in the Swift environment.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY="${OPENETA_TILELANG_OVERLAY:-${REPO_ROOT}/.runtime/tilelang-0.1.14-py312}"
UV_BIN="${UV_BIN:-uv}"
PYTHON="${PYTHON:-${SWIFT_VENV:-${REPO_ROOT}/.venv-embodied-swift}/bin/python}"
INDEX_URL="${INDEX_URL:-https://pypi.org/simple}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  echo "uv is required to install the TileLang overlay" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Swift Python not found: ${PYTHON}" >&2
  exit 2
fi

mkdir -p "${OVERLAY}"
"${UV_BIN}" pip install \
  --target "${OVERLAY}" \
  --no-deps \
  --index-url "${INDEX_URL}" \
  'tilelang==0.1.14' \
  'apache-tvm-ffi==0.1.12' \
  'ml-dtypes==0.6.0' \
  'z3-solver==4.15.4.0'

PYTHONPATH="${OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" -c \
  'import importlib.metadata as m, tilelang; print("TileLang overlay ready:", m.version("tilelang"))'
