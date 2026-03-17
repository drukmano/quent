#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <new-version>"
  echo "  e.g. $0 5.1.0"
  exit 1
fi

NEW_VERSION="$1"

# Validate semver format: MAJOR.MINOR.PATCH (digits only)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: version must match MAJOR.MINOR.PATCH (e.g. 5.1.0). Got: '$NEW_VERSION'"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYPROJECT="$REPO_ROOT/pyproject.toml"
SPEC="$REPO_ROOT/SPEC.md"
TODAY="$(date +%Y-%m-%d)"

echo "==> Bumping version to $NEW_VERSION (date: $TODAY)"

# Update pyproject.toml: replace the version = "..." line
sed -i '' "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$PYPROJECT"

# Update SPEC.md: replace **Version:** X.Y.Z | **Date:** YYYY-MM-DD
sed -i '' "s/\*\*Version:\*\* [0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]* | \*\*Date:\*\* [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/**Version:** $NEW_VERSION | **Date:** $TODAY/" "$SPEC"

echo "==> Running uv lock"
(cd "$REPO_ROOT" && uv lock)

echo "==> Changes:"
git -C "$REPO_ROOT" diff

echo ""
read -r -p "Commit these changes? [y/n] " ANSWER
if [[ "$ANSWER" == "y" || "$ANSWER" == "Y" ]]; then
  git -C "$REPO_ROOT" add "$PYPROJECT" "$SPEC" "$REPO_ROOT/uv.lock"
  git -C "$REPO_ROOT" commit -m "bump version to $NEW_VERSION"
  echo "==> Committed: bump version to $NEW_VERSION"
else
  echo "==> Skipped commit."
fi
