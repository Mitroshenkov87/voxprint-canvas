; Voxprint AI Canvas
; Shared Start-menu folder: Voxprint
; Windows 11 24H2 or newer, 64-bit. The installer does not download model weights.
; It records the VRAM tier with: python -m canvas.models.tiers

#define MyAppName "Voxprint AI Canvas"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Aleksandr Mitroshenkov"
#define MyAppExeName "voxprint-canvas.exe"

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
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "staging\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Run]
Filename: "{cmd}"; Parameters: "/C python -m canvas.models.tiers > ""{localappdata}\Voxprint\Canvas\tier.txt"""; Flags: runhidden waituntilterminated; StatusMsg: "Checking the GPU tier"
Filename: "{cmd}"; Parameters: "/C python -m canvas.app.main --register-models"; Flags: runhidden waituntilterminated; StatusMsg: "Registering the shared models folder"

[UninstallRun]
Filename: "{cmd}"; Parameters: "/C python -m canvas.app.main --unregister-models"; Flags: runhidden waituntilterminated
