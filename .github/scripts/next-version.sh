#!/usr/bin/env bash
# The version this run builds, and whether it publishes it, as GitHub outputs
# (version=, published=). Semantic versioning from conventional commits, as
# the engine's releases: since the last v tag, a breaking change (`type!:` or
# BREAKING CHANGE) raises the minor while below 1.0 and the major after, a
# `feat:` the minor, a `fix:` or `perf:` the patch, and anything else
# releases nothing (the build keeps the last release's version). A push to
# dev with one publishes; a pushed v tag publishes its own version.
set -euo pipefail

if [[ "${GITHUB_REF_TYPE:-}" == tag ]]; then
  echo "version=${GITHUB_REF_NAME#v}"
  echo "published=true"
  exit 0
fi

last=$(git describe --tags --abbrev=0 --match 'v[0-9]*.[0-9]*.[0-9]*' 2>/dev/null || echo v0.0.0)
IFS=. read -r major minor patch <<< "${last#v}"
bump=none
while IFS= read -r -d $'\x1e' message; do
  message=${message#$'\n'}
  header=${message%%$'\n'*}
  if [[ "$header" =~ ^[a-z]+(\([^\)]*\))?!: ]] || [[ "$message" == *"BREAKING CHANGE"* ]]; then
    bump=breaking
    break
  elif [[ "$header" =~ ^feat(\([^\)]*\))?: ]]; then
    bump=feat
  elif [[ "$header" =~ ^(fix|perf)(\([^\)]*\))?: ]] && [[ "$bump" == none ]]; then
    bump=fix
  fi
done < <(git log --format='%B%x1e' "$last..HEAD")

case "$bump" in
  breaking)
    if (( major == 0 )); then minor=$((minor + 1)); else major=$((major + 1)); minor=0; fi
    patch=0 ;;
  feat) minor=$((minor + 1)); patch=0 ;;
  fix) patch=$((patch + 1)) ;;
esac

published=false
if [[ "$bump" != none && "${GITHUB_REF:-}" == refs/heads/dev && "${GITHUB_EVENT_NAME:-}" == push ]]; then
  published=true
fi
echo "version=$major.$minor.$patch"
echo "published=$published"
