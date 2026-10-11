; Voxprint AI Canvas
; Shared Start-menu folder: Voxprint
; Windows 11 24H2 or newer, 64-bit.
; The setup copies the app. It does not embed GPU libraries or model files.
; install.ps1 installs missing dependencies, including the default image pack, and runs: python -m canvas.models.tiers

#define MyAppName "Voxprint AI Canvas"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Aleksandr Mitroshenkov"

[Setup]
AppId={{8F3C1A2E-6B44-4C1A-9E2D-7A0C5B1E4D33}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Voxprint\Canvas
DefaultGroupName=Voxprint
DisableProgramGroupPage=no
OutputBaseFilename=VoxprintCanvasSetup
Compression=lzma2
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.26100

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "staging\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\voxprint-canvas.cmd"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\install.ps1"""; Flags: waituntilterminated; StatusMsg: "Setting up Voxprint AI Canvas"

[UninstallRun]
Filename: "{app}\venv\Scripts\python.exe"; Parameters: "-m canvas.app.main --unregister-models"; Flags: runhidden waituntilterminated
