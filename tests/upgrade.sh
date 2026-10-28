#!/bin/sh
set -eu
[ "$#" -eq 1 ] || { echo 'Usage: sh tests/upgrade.sh ARCHIVE.tar.gz' >&2; exit 2; }
archive=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
work=$(mktemp -d "${TMPDIR:-/tmp}/tide-motion-upgrade-test-XXXXXXXX")
# Keep artifacts for diagnosing a failure; no user installation is modified.
root="$work/with spaces and 'quotes'"
mkdir -p "$work/bootstrap" "$root/models"
tar -xzf "$archive" -C "$work/bootstrap"
tool=$(find "$work/bootstrap" -type f -name tide-motion | head -n 1)
[ -n "$tool" ]
printf 'keep model\n' > "$root/models/keep.txt"
printf 'keep settings\n' > "$root/settings.json"
"$tool" upgrade --archive "$archive" --check --root "$work/untouched"
[ ! -e "$work/untouched" ]
"$tool" upgrade --archive "$archive" --root "$root" --force
tool="$root/bin/tide-motion"
"$tool" upgrade --archive "$archive" --force
[ "$(cat "$root/models/keep.txt")" = 'keep model' ]
[ "$(cat "$root/settings.json")" = 'keep settings' ]
for item in install.sh install.ps1 README.md PROMPTING.md licenses/META-LLAMA-3.txt; do [ -f "$root/$item" ]; done
# A symlink at a late destination triggers rollback of the preceding swaps.
printf 'rollback README\n' > "$root/README.md"
printf 'rollback note\n' > "$root/notes/custom-sentinel.txt"
mv "$root/LICENSE" "$work/original-license"
ln -s "$work/original-license" "$root/LICENSE"
if "$tool" upgrade --archive "$archive" --force > "$work/rejected" 2>&1; then echo 'Expected rollback refusal' >&2; exit 1; fi
grep -q 'symlink' "$work/rejected"
[ "$(cat "$root/README.md")" = 'rollback README' ]
[ "$(cat "$root/notes/custom-sentinel.txt")" = 'rollback note' ]
rm "$root/LICENSE"
mv "$work/original-license" "$root/LICENSE"
mkdir "$work/bad"
bad="$work/bad/$(basename "$archive")"
cp "$archive" "$bad"
printf '%064d\n' 0 > "$bad.sha256"
if "$tool" upgrade --archive "$bad" --force > "$work/rejected" 2>&1; then echo 'Accepted bad checksum' >&2; exit 1; fi
grep -q 'checksum mismatch' "$work/rejected"
mkdir "$work/unsafe"
ln -s /tmp "$work/unsafe/escape"
tar -czf "$bad" -C "$work/unsafe" escape
if command -v sha256sum >/dev/null 2>&1; then sha256sum "$bad" | cut -d ' ' -f 1 > "$bad.sha256"; else shasum -a 256 "$bad" | cut -d ' ' -f 1 > "$bad.sha256"; fi
if "$tool" upgrade --archive "$bad" --force > "$work/rejected" 2>&1; then echo 'Accepted escaping symlink' >&2; exit 1; fi
grep -q 'Unsafe archive' "$work/rejected"
"$tool" --version
echo "Offline/live upgrade, preservation, rollback and refusal checks passed: $work"
