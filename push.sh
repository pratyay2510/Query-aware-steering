#!/usr/bin/env bash
# Push local changes to GitHub. Prompts for a commit message.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

git add -A

if git diff --cached --quiet; then
  echo "Nothing new to commit — pushing any existing commits."
else
  read -rp "Commit message: " msg
  msg="${msg:-update}"
  git commit -m "$msg"
fi

git push -u origin main
