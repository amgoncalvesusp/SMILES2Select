#!/usr/bin/env bash
# This header is followed by the checksummed frozen application archive.
set -euo pipefail
version='@S2S_VERSION@'
expected_sha256='@S2S_SHA256@'
if [[ ${1:-} == --help ]]; then
    printf 'SMILES2Select %s — per-user Linux installer\n' "$version"
    printf '%s\n' 'Run without sudo: bash SMILES2Select-Setup-*.run' \
        'Installs under ${XDG_DATA_HOME:-$HOME/.local/share}/smiles2select.' \
        'Launcher: ${XDG_BIN_HOME:-$HOME/.local/bin}/smiles2select.' \
        'Uninstall: bash <installation directory>/uninstall.sh' \
        'User settings and sessions outside the installation directory are preserved.'
    exit 0
fi
[[ $# == 0 ]] || { echo 'Unknown option. Use --help.' >&2; exit 2; }
[[ $(uname -m) == x86_64 ]] || { echo 'This installer requires x86_64 Linux.' >&2; exit 1; }
data_dir=${XDG_DATA_HOME:-"$HOME/.local/share"}
bin_dir=${XDG_BIN_HOME:-"$HOME/.local/bin"}
check_path() {
    [[ $1 == /* && $1 != *[[:cntrl:]=]* ]] || {
        echo 'Installation paths must be absolute, without control characters or =.' >&2; exit 1;
    }
}
check_path "$data_dir"
check_path "$bin_dir"
mkdir -p "$data_dir" "$bin_dir" "$data_dir/applications"
data_dir=$(realpath "$data_dir")
bin_dir=$(realpath "$bin_dir")
check_path "$data_dir"
check_path "$bin_dir"
app_dir="$data_dir/smiles2select"
launcher="$bin_dir/smiles2select"
desktop="$data_dir/applications/smiles2select.desktop"
marker='SMILES2Select per-user installer v1'
if [[ -L $app_dir || ( -e $app_dir && ! -d $app_dir ) ]]; then
    echo 'Installation directory is a symlink or is not a directory.' >&2; exit 1
fi
if [[ -d $app_dir ]] && [[ ! -f $app_dir/.installer-owned || $(cat "$app_dir/.installer-owned") != "$marker" ]]; then
    echo "Refusing to replace an unmanaged directory: $app_dir" >&2; exit 1
fi
if [[ -e $launcher || -L $launcher ]] && [[ ! -L $launcher || $(readlink "$launcher") != "$app_dir/SMILES2Select" ]]; then
    echo "Refusing to replace an unrelated launcher: $launcher" >&2; exit 1
fi
if [[ -e $desktop || -L $desktop ]] && { [[ -L $desktop ]] || ! grep -qx 'X-SMILES2Select-Installer=1' "$desktop"; }; then
    echo "Refusing to replace an unrelated desktop entry: $desktop" >&2; exit 1
fi
work=$(mktemp -d "$data_dir/.smiles2select-install.XXXXXX")
backup=''
published=false
completed=false
cleanup() {
    if [[ $completed == false && $published == true ]]; then
        if [[ -n $backup ]]; then
            rm -rf -- "$app_dir"
        else
            bash "$app_dir/uninstall.sh"
        fi
    fi
    if [[ -n $backup && -d $backup && ! -e $app_dir ]]; then mv -- "$backup" "$app_dir"; fi
    rm -rf -- "$work"
}
trap cleanup EXIT
payload_line=$(awk '/^__S2S_ARCHIVE_BELOW__$/ {print NR + 1; exit}' "$0")
tail -n +"$payload_line" "$0" > "$work/payload.tar.gz"
actual_sha256=$(sha256sum "$work/payload.tar.gz" | cut -d ' ' -f 1)
[[ $actual_sha256 == "$expected_sha256" ]] || { echo 'Installer integrity check failed.' >&2; exit 1; }
mkdir "$work/app"
tar --no-same-owner --no-same-permissions -xzf "$work/payload.tar.gz" -C "$work/app"
[[ -f "$work/app/SMILES2Select" ]] || { echo 'Installer executable missing.' >&2; exit 1; }
chmod +x "$work/app/SMILES2Select"
printf '%s\n' "$marker" > "$work/app/.installer-owned"
{
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'app_dir=%q\nlauncher=%q\ndesktop=%q\n' "$app_dir" "$launcher" "$desktop"
    cat "$work/app/.uninstall-template"
} > "$work/app/uninstall.sh"
rm "$work/app/.uninstall-template"
chmod +x "$work/app/uninstall.sh"
# Desktop Exec quoting follows the desktop entry specification, not shell quoting.
exec_path=${app_dir//\\/\\\\}
exec_path=${exec_path//\"/\\\"}
exec_path=${exec_path//\$/\\\$}
exec_path=${exec_path//\`/\\\`}
exec_path=${exec_path//%/%%}
exec_path=${exec_path//\\/\\\\}
icon_path=${app_dir//\\/\\\\}
cat > "$work/smiles2select.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=SMILES2Select
Comment=Molecular selection and S2S-Decision models
Exec="$exec_path/SMILES2Select"
Icon=$icon_path/_internal/smiles2select/assets/SMILES2Select.png
Terminal=false
Categories=Science;Chemistry;
X-SMILES2Select-Installer=1
EOF
if [[ -d $app_dir ]]; then
    backup="$work/previous"
    mv -- "$app_dir" "$backup"
fi
mv -- "$work/app" "$app_dir"
published=true
ln -s "$app_dir/SMILES2Select" "$work/launcher"
mv -Tf -- "$work/launcher" "$launcher"
mv -f -- "$work/smiles2select.desktop" "$desktop"
completed=true
printf 'SMILES2Select %s installed in %s\nLaunch: %s\nUninstall: bash "%s/uninstall.sh"\n' \
    "$version" "$app_dir" "$launcher" "$app_dir"
exit 0
__S2S_ARCHIVE_BELOW__
