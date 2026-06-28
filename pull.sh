#!/usr/bin/env bash
# Pull the latest changes from GitHub.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git pull --ff-only origin main
