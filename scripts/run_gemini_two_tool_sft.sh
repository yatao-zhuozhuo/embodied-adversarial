#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CODINGROUTER_API_KEY:?CODINGROUTER_API_KEY must be set}"
export OPENETA_TWO_TOOL_MCP_LOG_DIR="${OPENETA_TWO_TOOL_MCP_LOG_DIR:-${REPO_ROOT}/SFT_data/reports/gemini_xml_v1}"
exec bash "${REPO_ROOT}/scripts/run_two_tool_sft_collection.sh" --provider gemini "$@"
