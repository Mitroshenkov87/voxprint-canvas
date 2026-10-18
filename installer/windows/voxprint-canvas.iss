; VOXPrint AI Canvas
; Install folder: C:\Program Files\VOXPrint AI Suite\Canvas\
; Start menu folder: VOXPrint AI Suite
; GUI: voxprint-canvas.exe  CLI: voxprint-canvas-cli.exe
; voxprint-canvas.cmd remains beside the program for the first-run bootstrap.
; Windows 11 24H2 or newer, 64-bit.
; The setup copies the app. It does not embed GPU libraries or model files.
; install.ps1 installs missing dependencies, including the default image pack, and runs: python -m canvas.models.tiers
; CurStepChanged runs the machine scope elevated, then the user scope as the original user.
; Setup exits non-zero when either scope fails. Uninstall stays elevated and does not drop privileges.

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

[Code]
procedure ExitProcess(uExitCode: Cardinal);
  external 'ExitProcess@kernel32.dll stdcall';

function RunSetup(const Params: String; const DropPrivileges: Boolean): Integer;
var
  ResultCode: Integer;
  Started: Boolean;
begin
  Result := 0;
  if DropPrivileges then
  begin
    Started := ExecAsOriginalUser(
      ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
      Params,
      ExpandConstant('{app}'),
      SW_SHOW,
      ewWaitUntilTerminated,
      ResultCode
    );
  end
  else
  begin
    Started := Exec(
      ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
      Params,
      ExpandConstant('{app}'),
      SW_SHOW,
      ewWaitUntilTerminated,
      ResultCode
    );
  end;
  if not Started then
  begin
    SuppressibleMsgBox('VOXPrint AI Canvas could not start its setup script.', mbError, MB_OK, IDOK);
    Result := 1;
    Exit;
  end;
  if ResultCode <> 0 then
  begin
    SuppressibleMsgBox('VOXPrint AI Canvas setup failed. The machine log is ' + ExpandConstant('{app}\setup-work\canvas-install.log') + '. The user log is ' + ExpandConstant('{localappdata}\VOXPrint AI Suite\logs\canvas-install.log') + '.', mbError, MB_OK, IDOK);
    if (ResultCode > 0) and (ResultCode <= 255) then
      Result := ResultCode
    else
      Result := 1;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Code, CleanupCode: Integer;
begin
  if CurStep <> ssPostInstall then
    Exit;
  Code := RunSetup(ExpandConstant('-NoProfile -ExecutionPolicy Bypass -File "{app}\install.ps1" -Scope Machine'), False);
  if Code = 0 then
    Code := RunSetup(ExpandConstant('-NoProfile -ExecutionPolicy Bypass -File "{app}\install.ps1" -Scope User'), True);
  CleanupCode := RunSetup(ExpandConstant('-NoProfile -ExecutionPolicy Bypass -File "{app}\install.ps1" -Scope Cleanup'), False);
  if Code = 0 then
    Code := CleanupCode;
  if Code <> 0 then
    ExitProcess(Code);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ResultCode: Integer;
begin
  if CurUninstallStep <> usUninstall then
    Exit;
  Exec(
    ExpandConstant('{app}\venv\Scripts\python.exe'),
    '-I -m canvas.app.main --unregister-models',
    ExpandConstant('{app}'),
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  );
end;
