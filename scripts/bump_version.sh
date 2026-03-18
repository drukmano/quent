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
CHANGELOG="$REPO_ROOT/CHANGELOG.md"
DOCS_CHANGELOG="$REPO_ROOT/docs/changelog.md"
TODAY="$(date +%Y-%m-%d)"

echo "==> Bumping version to $NEW_VERSION (date: $TODAY)"

# Update pyproject.toml: replace the version = "..." line
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$PYPROJECT" && rm -f "$PYPROJECT.bak"

# Update SPEC.md: replace **Version:** X.Y.Z | **Date:** YYYY-MM-DD
sed -i.bak "s/\*\*Version:\*\* [0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]* | \*\*Date:\*\* [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/**Version:** $NEW_VERSION | **Date:** $TODAY/" "$SPEC" && rm -f "$SPEC.bak"

# Insert changelog placeholder for the new version before the first ## [...] section
# Uses awk for portable "insert before first match" (POSIX-compatible, works on macOS and Linux)
PLACEHOLDER="## [$NEW_VERSION] - $TODAY\n\n### Added\n\n### Changed\n\n### Fixed\n"
for CLOG in "$CHANGELOG" "$DOCS_CHANGELOG"; do
  awk -v placeholder="$PLACEHOLDER" '
    !done && /^## \[/ {
      printf "%s\n", placeholder
      done = 1
    }
    { print }
  ' "$CLOG" > "$CLOG.tmp" && mv "$CLOG.tmp" "$CLOG"
done

echo "==> Running uv lock"
(cd "$REPO_ROOT" && uv lock)

echo "==> Changes:"
git -C "$REPO_ROOT" diff

echo ""
read -r -p "Commit these changes? [y/n] " ANSWER
if [[ "$ANSWER" == "y" || "$ANSWER" == "Y" ]]; then
  git -C "$REPO_ROOT" add "$PYPROJECT" "$SPEC" "$CHANGELOG" "$DOCS_CHANGELOG" "$REPO_ROOT/uv.lock"
  git -C "$REPO_ROOT" commit -m "bump version to $NEW_VERSION"
  echo "==> Committed: bump version to $NEW_VERSION"
else
  echo "==> Skipped commit."
fi
