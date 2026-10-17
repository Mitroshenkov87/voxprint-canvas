; VOXPrint AI Canvas
; Install folder: C:\Program Files\VOXPrint AI Suite\Canvas\
; Start menu folder: VOXPrint AI Suite
; GUI: voxprint-canvas.exe  CLI: voxprint-canvas-cli.exe
; voxprint-canvas.cmd remains beside the program for the first-run bootstrap.
; Windows 11 24H2 or newer, 64-bit.
; The setup copies the app. It does not embed GPU libraries or model files.
; install.ps1 installs missing dependencies, including the default image pack, and runs: python -m canvas.models.tiers
; CurStepChanged runs that script as the user who started Setup, then Setup exits non-zero when it fails.
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

[UninstallRun]
Filename: "{app}\venv\Scripts\python.exe"; Parameters: "-m canvas.app.main --unregister-models"; Flags: runhidden waituntilterminated runasoriginaluser

[Code]
procedure ExitProcess(uExitCode: Cardinal);
  external 'ExitProcess@kernel32.dll stdcall';

function OriginalAccount: String;
var
  ResultCode: Integer;
  Path, Command: String;
  Text: AnsiString;
begin
  Path := ExpandConstant('{tmp}\voxprint-account.txt');
  Command := '/d /c "' + ExpandConstant('{sys}\whoami.exe') + '" > "' + Path + '"';
  if not ExecAsOriginalUser(ExpandConstant('{sys}\cmd.exe'), Command, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    Result := '';
    Exit;
  end;
  if (ResultCode <> 0) or (not LoadStringFromFile(Path, Text)) then
  begin
    Result := '';
    Exit;
  end;
  Result := Trim(String(Text));
end;

procedure GrantInstallFolder(const Account: String);
var
  ResultCode: Integer;
  Params: String;
begin
  Params := ExpandConstant('"{app}"') + ' /grant "' + Account + ':(OI)(CI)M" /C';
  if (not Exec(ExpandConstant('{sys}\icacls.exe'), Params, '', SW_HIDE, ewWaitUntilTerminated, ResultCode)) or (ResultCode <> 0) then
  begin
    SuppressibleMsgBox('VOXPrint AI Canvas could not prepare its install folder.', mbError, MB_OK, IDOK);
    ExitProcess(1);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Started: Boolean;
  Account: String;
begin
  if CurStep <> ssPostInstall then
    Exit;
  Account := OriginalAccount;
  if Account = '' then
  begin
    SuppressibleMsgBox('VOXPrint AI Canvas could not identify the account that started Setup.', mbError, MB_OK, IDOK);
    ExitProcess(1);
  end;
  GrantInstallFolder(Account);
  Started := ExecAsOriginalUser(
    ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
    ExpandConstant('-NoProfile -ExecutionPolicy Bypass -File "{app}\install.ps1"'),
    '',
    SW_SHOW,
    ewWaitUntilTerminated,
    ResultCode
  );
  if not Started then
  begin
    SuppressibleMsgBox('VOXPrint AI Canvas could not start its setup script.', mbError, MB_OK, IDOK);
    ExitProcess(1);
  end;
  if ResultCode <> 0 then
  begin
    SuppressibleMsgBox('VOXPrint AI Canvas setup failed. The log is %LOCALAPPDATA%\Voxprint\logs\canvas-install.log.', mbError, MB_OK, IDOK);
    if (ResultCode > 0) and (ResultCode <= 255) then
      ExitProcess(ResultCode)
    else
      ExitProcess(1);
  end;
end;
