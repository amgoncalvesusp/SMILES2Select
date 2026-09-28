#!/usr/bin/env bash
# Wrap an existing PyInstaller Linux bundle in a per-user installer.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "Usage: bash packaging/build_linux.sh BUNDLE VERSION OUTPUT_DIR" >&2
    exit 2
fi
bundle=$(realpath "$1")
version=$2
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][a-zA-Z0-9.]+)?$ ]] || exit 2
[[ $(uname -m) == x86_64 ]] || { echo 'Build on x86_64 Linux.' >&2; exit 1; }
[[ -x "$bundle/SMILES2Select" ]] || { echo 'Frozen executable missing.' >&2; exit 1; }
mkdir -p "$3"
output=$(realpath "$3")
scripts=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
cp -a "$bundle/." "$work/bundle"
cp "$scripts/../LICENSE" "$scripts/../README.md" "$scripts/../CHANGELOG.md" "$work/bundle/"
cp "$scripts/linux_uninstall.sh" "$work/bundle/.uninstall-template"
tar -czf "$work/payload.tar.gz" -C "$work/bundle" .
digest=$(sha256sum "$work/payload.tar.gz" | cut -d ' ' -f 1)
installer="$output/SMILES2Select-Setup-$version-linux-x86_64.run"
sed -e "s/@S2S_VERSION@/$version/g" -e "s/@S2S_SHA256@/$digest/g" \
    "$scripts/linux_install.sh" > "$installer"
cat "$work/payload.tar.gz" >> "$installer"
chmod +x "$installer"
echo "$installer"
