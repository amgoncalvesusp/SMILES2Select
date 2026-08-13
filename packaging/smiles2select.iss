; Inno Setup definition for the portable PyInstaller onedir bundle.
; Build the bundle first, then compile this file with:
;   ISCC.exe /DAppVersion=3.0.1 /DSourceDir=...\dist\SMILES2Select packaging\smiles2select.iss

#ifndef AppVersion
  #define AppVersion "3.0.1"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\SMILES2Select"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist"
#endif

[Setup]
AppId={{B1D0B6C7-6D51-4F18-9D15-6D9CEB390300}
AppName=SMILES2Select
AppVersion={#AppVersion}
AppVerName=SMILES2Select {#AppVersion}
AppPublisher=Adriano Marques Goncalves - UNIARA
AppPublisherURL=https://github.com/amgoncalvesusp/SMILES2Select
AppSupportURL=https://github.com/amgoncalvesusp/SMILES2Select/issues
AppUpdatesURL=https://github.com/amgoncalvesusp/SMILES2Select/releases
DefaultDirName={localappdata}\Programs\SMILES2Select
DefaultGroupName=SMILES2Select
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=SMILES2Select-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=SMILES2Select {#AppVersion}
Uninstallable=yes
ChangesAssociations=no
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; This recursive entry is the completeness boundary: every PyInstaller output
; file, including .dll, .pyd, Qt plugins and RDKit data, is installed.
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\CHANGELOG.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\SMILES2Select"; Filename: "{app}\SMILES2Select.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\SMILES2Select"; Filename: "{app}\SMILES2Select.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\SMILES2Select.exe"; Description: "Launch SMILES2Select"; Flags: postinstall skipifsilent nowait
