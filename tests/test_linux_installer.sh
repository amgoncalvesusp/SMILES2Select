#!/usr/bin/env bash
# Behavioral installer checks, without requiring Qt or a full frozen bundle.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
export HOME="$scratch/home with spaces"
export XDG_DATA_HOME="$HOME/data"
export XDG_BIN_HOME="$HOME/bin"
mkdir -p "$scratch/bundle/_internal/smiles2select/assets" "$HOME/.config"
printf '#!/usr/bin/env bash\nprintf "bundle-v1:%%s\\n" "$*"\n' > "$scratch/bundle/SMILES2Select"
chmod +x "$scratch/bundle/SMILES2Select"
cp "$repo/src/smiles2select/assets/SMILES2Select.png" "$scratch/bundle/_internal/smiles2select/assets/"
printf 'preserve me\n' > "$HOME/.config/smiles2select-user-data"
printf 'preserve session\n' > "$HOME/example.s2s.sqlite"
bash "$repo/packaging/build_linux.sh" "$scratch/bundle" 3.4.1 "$scratch/out"
installer="$scratch/out/SMILES2Select-Setup-3.4.1-linux-x86_64.run"
bash "$installer" --help > /dev/null
bash "$installer"
test "$("$XDG_BIN_HOME/smiles2select" 'test argument')" = 'bundle-v1:test argument'
test -f "$XDG_DATA_HOME/applications/smiles2select.desktop"
test -f "$XDG_DATA_HOME/smiles2select/uninstall.sh"
test -f "$XDG_DATA_HOME/smiles2select/LICENSE"
test -f "$XDG_DATA_HOME/smiles2select/README.md"
bash "$installer" # Same-version update must succeed.
test "$("$XDG_BIN_HOME/smiles2select" repeated)" = 'bundle-v1:repeated'
# A failed desktop-entry write must restore the previous working installation.
mkdir "$scratch/fail-bin"
cat > "$scratch/fail-bin/mv" <<'EOF'
#!/usr/bin/env bash
if [[ ${*: -1} == */applications/smiles2select.desktop ]]; then exit 1; fi
exec /usr/bin/mv "$@"
EOF
chmod +x "$scratch/fail-bin/mv"
if PATH="$scratch/fail-bin:$PATH" bash "$installer"; then
    echo 'Simulated failed update succeeded' >&2; exit 1
fi
test "$("$XDG_BIN_HOME/smiles2select" after-rollback)" = 'bundle-v1:after-rollback'
cp "$installer" "$scratch/corrupt.run"
printf 'corrupt' >> "$scratch/corrupt.run"
if bash "$scratch/corrupt.run"; then
    echo 'Corrupted payload was accepted' >&2; exit 1
fi
test "$("$XDG_BIN_HOME/smiles2select" after-corruption)" = 'bundle-v1:after-corruption'
# Uninstall uses recorded paths even if the environment changes.
XDG_DATA_HOME="$scratch/other" XDG_BIN_HOME="$scratch/other-bin" bash "$XDG_DATA_HOME/smiles2select/uninstall.sh"
test ! -e "$XDG_DATA_HOME/smiles2select"
test ! -e "$XDG_BIN_HOME/smiles2select"
test ! -e "$XDG_DATA_HOME/applications/smiles2select.desktop"
test -f "$HOME/.config/smiles2select-user-data"
test -f "$HOME/example.s2s.sqlite"
if PATH="$scratch/fail-bin:$PATH" bash "$installer"; then
    echo 'Simulated failed first install succeeded' >&2; exit 1
fi
test ! -e "$XDG_DATA_HOME/smiles2select"
test ! -L "$XDG_BIN_HOME/smiles2select"
printf 'unrelated launcher' > "$XDG_BIN_HOME/smiles2select"
if bash "$installer"; then
    echo 'Unrelated launcher was overwritten' >&2; exit 1
fi
test "$(cat "$XDG_BIN_HOME/smiles2select")" = 'unrelated launcher'
rm "$XDG_BIN_HOME/smiles2select"
ln -s "$HOME/.config" "$XDG_DATA_HOME/smiles2select"
if bash "$installer"; then
    echo 'Symlink installation directory was accepted' >&2; exit 1
fi
test -f "$HOME/.config/smiles2select-user-data"
rm "$XDG_DATA_HOME/smiles2select"
mkdir -p "$XDG_DATA_HOME/smiles2select"
printf 'unrelated' > "$XDG_DATA_HOME/smiles2select/keep"
if bash "$installer"; then
    echo 'Unowned directory was overwritten' >&2; exit 1
fi
test "$(cat "$XDG_DATA_HOME/smiles2select/keep")" = unrelated
echo 'Linux installer: install, update, launch, rollback, integrity, uninstall and data preservation passed.'
