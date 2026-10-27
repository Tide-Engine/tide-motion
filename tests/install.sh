#!/bin/sh
set -eu
[ "$#" -eq 1 ] || { echo 'Usage: sh tests/install.sh ARCHIVE.tar.gz' >&2; exit 2; }
repo=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
archive=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
work=$(mktemp -d "${TMPDIR:-/tmp}/tide-motion-install-XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT HUP INT TERM
root="$work/with spaces and 'quotes'"
mkdir -p "$root/models"
echo keep > "$root/models/keep.txt"
sh "$repo/install.sh" "$archive" "$root"
sh "$repo/install.sh" "$archive" "$root"
[ "$(cat "$root/models/keep.txt")" = keep ]
for item in examples/jobs.json notes/implementation.md LICENSE; do [ -f "$root/$item" ]; done
env -u LD_LIBRARY_PATH -u DYLD_LIBRARY_PATH "$root/bin/tide-motion" --version
"$root/bin/tide-motion" --help >/dev/null
# Corrupt checksums and escaping links must fail before replacing the executable.
cp "$archive" "$work/bad.tar.gz"
printf '%064d\n' 0 > "$work/bad.tar.gz.sha256"
if sh "$repo/install.sh" "$work/bad.tar.gz" "$root" > "$work/rejected" 2>&1; then
    echo 'Installer accepted an invalid checksum' >&2; exit 1
fi
grep -q 'checksum mismatch' "$work/rejected"
mkdir "$work/unsafe"
ln -s /tmp "$work/unsafe/escape"
tar -czf "$work/unsafe.tar.gz" -C "$work/unsafe" escape
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$work/unsafe.tar.gz" > "$work/unsafe.tar.gz.sha256"
else
    shasum -a 256 "$work/unsafe.tar.gz" > "$work/unsafe.tar.gz.sha256"
fi
if sh "$repo/install.sh" "$work/unsafe.tar.gz" "$root" > "$work/rejected" 2>&1; then
    echo 'Installer accepted an escaping symlink' >&2; exit 1
fi
grep -q 'Unsafe archive' "$work/rejected"
env -u LD_LIBRARY_PATH -u DYLD_LIBRARY_PATH "$root/bin/tide-motion" --version
echo 'Installer checks passed'
