#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible name retained for existing project documentation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/setup_network.sh"