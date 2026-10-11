; VOXPrint AI Canvas
; Install folder: C:\Program Files\VOXPrint AI Suite\Canvas\
; Start menu folder: VOXPrint AI Suite
; GUI: voxprint-canvas.exe  CLI: voxprint-canvas-cli.exe
; voxprint-canvas.cmd remains beside the program for the first-run bootstrap.
; Windows 11 24H2 or newer, 64-bit.
; The setup copies the app. It does not embed GPU libraries or model files.
; install.ps1 installs missing dependencies, including the default image pack, and runs: python -m canvas.models.tiers
; An upgrade keeps the AppId, ignores the previous directory, and the setup script removes the old folder and shortcuts.

#define MyAppName "VOXPrint AI Canvas"
#define MySuiteName "VOXPrint AI Suite"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Aleksandr Mitroshenkov"

[Setup]
AppId={{8F3C1A2E-6B44-4C1A-9E2D-7A0C5B1E4D33}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MySuiteName}\Canvas
DefaultGroupName={#MySuiteName}
DisableDirPage=yes
DisableProgramGroupPage=yes
UsePreviousAppDir=no
UsePreviousGroup=no
OutputBaseFilename=VoxprintCanvasSetup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.26100

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "staging\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\venv\Scripts\voxprint-canvas.exe"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"""; Flags: runasoriginaluser waituntilterminated; StatusMsg: "Setting up {#MyAppName}"

[UninstallRun]
Filename: "{app}\venv\Scripts\python.exe"; Parameters: "-m canvas.app.main --unregister-models"; Flags: runhidden waituntilterminated
