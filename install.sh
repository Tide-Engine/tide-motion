#!/bin/sh
# Install a downloaded release archive beside its .sha256 file, without sudo.
set -eu
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: sh install.sh ARCHIVE.tar.gz [INSTALL_ROOT]" >&2
    exit 2
fi
archive=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
root=${2:-"$HOME/.local/share/tide-motion"}
fail() { echo "$*" >&2; exit 1; }
[ -f "$archive.sha256" ] || fail "Missing checksum file: $archive.sha256"
expected=$(awk 'NR == 1 { print $1 }' "$archive.sha256")
[ "${#expected}" -eq 64 ] || fail 'Malformed archive checksum'
case "$expected" in *[!0-9a-fA-F]*) fail 'Malformed archive checksum';; esac
if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$archive" | awk '{ print $1 }')
else
    actual=$(shasum -a 256 "$archive" | awk '{ print $1 }')
fi
[ "$(printf '%s' "$expected" | tr A-F a-f)" = "$actual" ] || fail 'Archive checksum mismatch'
mkdir -p "$root"
root=$(cd "$root" && pwd -P)
stage=$(mktemp -d "$root/.install-XXXXXXXX")
backup=$stage/previous-bin
cleanup() {
    # Both paths are absolute and this scratch directory was created under root.
    case "$stage" in "$root"/.install-*) rm -rf -- "$stage";; esac
}
trap cleanup EXIT HUP INT TERM
tar -tzf "$archive" > "$stage/members"
while IFS= read -r member; do
    case "$member" in /*|*\\*|../*|*/../*|*/..|..) fail "Unsafe archive entry: $member";; esac
done < "$stage/members"
# CPack may preserve ONNX's relative library symlinks. Only simple sibling links
# are allowed; reject hard links, devices and links that could escape extraction.
tar -tvzf "$archive" > "$stage/types"
awk '
    substr($0, 1, 1) == "-" || substr($0, 1, 1) == "d" { next }
    substr($0, 1, 1) == "l" && $(NF-1) == "->" && $NF ~ /^libonnxruntime[.][A-Za-z0-9.]+$/ { next }
    { exit 1 }
' "$stage/types" || fail 'Unsafe archive entry type or link'
mkdir "$stage/payload"
tar -xzf "$archive" -C "$stage/payload"
payload=$stage/payload
if [ ! -f "$payload/bin/tide-motion" ]; then
    set -- "$payload"/*
    [ "$#" -eq 1 ] && [ -d "$1" ] || fail 'Incomplete Tide Motion package'
    payload=$1
fi
for required in bin/tide-motion licenses/META-LLAMA-3.txt examples/jobs.json notes/implementation.md README.md PROMPTING.md LICENSE; do
    [ -f "$payload/$required" ] || fail "Incomplete Tide Motion package: $required"
done
"$payload/bin/tide-motion" --version
if [ -e "$root/bin" ]; then mv "$root/bin" "$backup"; fi
if ! mv "$payload/bin" "$root/bin"; then
    if [ -d "$backup" ]; then mv "$backup" "$root/bin"; fi
    fail 'Could not replace the executable'
fi
for item in licenses examples notes README.md PROMPTING.md LICENSE; do
    cp -R "$payload/$item" "$root/"
done
echo "Installed Tide Motion in $root"
echo "Add this directory to PATH: $root/bin"
echo "Run: \"$root/bin/tide-motion\" --help"
