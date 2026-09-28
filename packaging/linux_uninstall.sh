# Paths are recorded by the installer so changed XDG settings cannot redirect removal.
marker='SMILES2Select per-user installer v1'
if [[ ${app_dir##*/} != smiles2select || -L $app_dir || ! -f $app_dir/.installer-owned || $(cat "$app_dir/.installer-owned") != "$marker" ]]; then
    echo 'Refusing to uninstall an unmanaged directory.' >&2; exit 1
fi
if [[ -L $launcher && $(readlink "$launcher") == "$app_dir/SMILES2Select" ]]; then
    rm -- "$launcher"
fi
if [[ ! -L $desktop && -f $desktop ]] && grep -qx 'X-SMILES2Select-Installer=1' "$desktop"; then
    rm -- "$desktop"
fi
rm -rf -- "$app_dir"
echo 'SMILES2Select uninstalled. User settings and sessions outside its installation directory were preserved.'
