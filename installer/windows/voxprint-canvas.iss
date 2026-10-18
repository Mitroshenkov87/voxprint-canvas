; VOXPrint AI Canvas
; Install folder: C:\Program Files\VOXPrint AI Suite\Canvas\
; Start menu folder: VOXPrint AI Suite
; GUI: voxprint-canvas.exe  CLI: voxprint-canvas-cli.exe
; voxprint-canvas.cmd remains beside the program for the first-run bootstrap.
; Windows 11 24H2 or newer, 64-bit.
; The setup copies the app. It does not embed GPU libraries or model files.
; install.ps1 installs missing dependencies, including the default image pack, and runs: python -m canvas.models.tiers
; The setup scripts run under PowerShell 7 (pwsh.exe). When it is missing, an interactive Setup offers
; to install the pinned official MSI from Microsoft (SHA-256 and Authenticode checked first);
; on No, on any failure, or in a silent Setup, Windows PowerShell 5.1 runs them instead.
; CurStepChanged runs the machine scope elevated, then the user scope as the original user.
; Setup exits non-zero when either scope fails. Uninstall stays elevated and does not drop privileges.

#define MyAppName "VOXPrint AI Canvas"
#define MySuiteName "VOXPrint AI Suite"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Aleksandr Mitroshenkov"

; Build the signature-check helper with the .NET Framework C# compiler that ships with Windows.
#define Csc AddBackslash(GetEnv("WINDIR")) + "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
#if !FileExists(Csc)
  #error The .NET Framework C# compiler was not found
#endif
#define CscExit Exec(Csc, '/nologo /optimize+ /target:exe /out:"' + AddBackslash(SourcePath) + 'vxsigcheck.exe" "' + AddBackslash(SourcePath) + 'vxsigcheck.cs"', SourcePath, 1, 0)
#if CscExit != 0
  #error vxsigcheck.cs did not compile
#endif

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
ShowLanguageDialog=auto

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[CustomMessages]
english.PwshAsk=PowerShell 7 is not installed. Install PowerShell 7 (Microsoft) now? Recommended.
russian.PwshAsk=PowerShell 7 не установлен. Установить PowerShell 7 (Microsoft) сейчас? Рекомендуется.
english.PwshInstalling=Installing PowerShell 7 (Microsoft)
russian.PwshInstalling=Установка PowerShell 7 (Microsoft)
english.PwshInstallingDesc=Downloading and checking the official installer from Microsoft.
russian.PwshInstallingDesc=Загрузка и проверка официального установщика Microsoft.
english.PwshFallback=Setup will continue with Windows PowerShell 5.1. Please install PowerShell 7 when possible: https://aka.ms/powershell-release?tag=stable
english.SetupScripts=Installing the program libraries and models. This can take a while; progress is written to the setup log.
russian.SetupScripts=Установка библиотек программы и моделей. Это может занять продолжительное время; ход установки записывается в журнал установки.
russian.PwshFallback=Установка продолжится с Windows PowerShell 5.1. По возможности установите PowerShell 7: https://aka.ms/powershell-release?tag=stable
english.GuideEnglish=VOXPrint AI Canvas Guide (English)
russian.GuideEnglish=VOXPrint AI Canvas Guide (English)
english.GuideRussian=Руководство VOXPrint AI Canvas (русский)
russian.GuideRussian=Руководство VOXPrint AI Canvas (русский)

[Files]
; staging\docs\VOXPrint-AI-Canvas-Guide-en.pdf and -ru.pdf land in {app}\docs.
Source: "staging\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "vxsigcheck.exe"; DestDir: "{tmp}"; Flags: dontcopy

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\venv\Scripts\voxprint-canvas.exe"; WorkingDir: "{app}"
Name: "{group}\{cm:GuideEnglish}"; Filename: "{app}\docs\VOXPrint-AI-Canvas-Guide-en.pdf"
Name: "{group}\{cm:GuideRussian}"; Filename: "{app}\docs\VOXPrint-AI-Canvas-Guide-ru.pdf"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Code]
const
  PwshMsiUrl = 'https://github.com/PowerShell/PowerShell/releases/download/v7.6.7/PowerShell-7.6.7-win-x64.msi';
  PwshMsiName = 'PowerShell-7.6.7-win-x64.msi';
  PwshMsiSha256 = 'ec575cef9ec56fcd905de0d5eeac9cb7f81a0a65abb87bb5cd0e87fdfe2b8cc0';
  PwshPublisher = 'Microsoft Corporation';
  PwshVersionsKey = 'SOFTWARE\Microsoft\PowerShellCore\InstalledVersions';

var
  ScriptHost: String;

procedure ExitProcess(uExitCode: Cardinal);
  external 'ExitProcess@kernel32.dll stdcall';

function FindPwsh: String;
var
  Versions: TArrayOfString;
  I: Integer;
  Location, Candidate: String;
begin
  Result := '';
  if RegGetSubkeyNames(HKLM64, PwshVersionsKey, Versions) then
  begin
    for I := 0 to GetArrayLength(Versions) - 1 do
    begin
      if (Result = '') and RegQueryStringValue(HKLM64, PwshVersionsKey + '\' + Versions[I], 'InstallLocation', Location) then
      begin
        Candidate := AddBackslash(Location) + 'pwsh.exe';
        if FileExists(Candidate) then
          Result := Candidate;
      end;
    end;
  end;
  if Result = '' then
  begin
    Candidate := ExpandConstant('{commonpf64}\PowerShell\7\pwsh.exe');
    if FileExists(Candidate) then
      Result := Candidate;
  end;
end;

function WindowsPowerShell: String;
begin
  Result := ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe');
end;

function InstallPwsh: Boolean;
var
  Page: TDownloadWizardPage;
  Msi, Helper: String;
  Code: Integer;
begin
  Result := False;
  Page := CreateDownloadPage(CustomMessage('PwshInstalling'), CustomMessage('PwshInstallingDesc'), nil);
  Page.Clear;
  Page.Add(PwshMsiUrl, PwshMsiName, PwshMsiSha256);
  Page.Show;
  try
    try
      Page.Download;
      Msi := ExpandConstant('{tmp}\') + PwshMsiName;
      ExtractTemporaryFile('vxsigcheck.exe');
      Helper := ExpandConstant('{tmp}\vxsigcheck.exe');
      Page.SetText(CustomMessage('PwshInstalling'), PwshMsiName);
      if (not Exec(Helper, '"' + Msi + '" "' + PwshPublisher + '"', '', SW_HIDE, ewWaitUntilTerminated, Code)) or (Code <> 0) then
        Log(Format('PowerShell 7 package failed the signature check (%d).', [Code]))
      else if (not Exec(ExpandConstant('{sys}\msiexec.exe'), '/i "' + Msi + '" /qn /norestart', '', SW_HIDE, ewWaitUntilTerminated, Code)) or ((Code <> 0) and (Code <> 3010)) then
        Log(Format('PowerShell 7 package did not install (%d).', [Code]))
      else
      begin
        ScriptHost := FindPwsh;
        Result := ScriptHost <> '';
        if not Result then
          Log('PowerShell 7 was installed but pwsh.exe was not found.');
      end;
    except
      Log('PowerShell 7 download failed: ' + GetExceptionMessage);
    end;
  finally
    Page.Hide;
  end;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  ScriptHost := FindPwsh;
  if ScriptHost = '' then
  begin
    if not WizardSilent then
    begin
      if MsgBox(CustomMessage('PwshAsk'), mbConfirmation, MB_YESNO) = IDYES then
        InstallPwsh;
    end;
    if ScriptHost = '' then
    begin
      ScriptHost := WindowsPowerShell;
      if not WizardSilent then
        MsgBox(CustomMessage('PwshFallback'), mbInformation, MB_OK);
    end;
  end;
  Log('Setup script host: ' + ScriptHost);
end;

function RunSetup(const Params: String; const DropPrivileges: Boolean): Integer;
var
  ResultCode: Integer;
  Started: Boolean;
begin
  Result := 0;
  if ScriptHost = '' then
    ScriptHost := FindPwsh;
  if ScriptHost = '' then
    ScriptHost := WindowsPowerShell;
  if DropPrivileges then
  begin
    Started := ExecAsOriginalUser(
      ScriptHost,
      Params,
      ExpandConstant('{app}'),
      SW_HIDE,
      ewWaitUntilTerminated,
      ResultCode
    );
  end
  else
  begin
    Started := Exec(
      ScriptHost,
      Params,
      ExpandConstant('{app}'),
      SW_HIDE,
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
  WizardForm.StatusLabel.Caption := CustomMessage('SetupScripts');
  WizardForm.ProgressGauge.Style := npbstMarquee;
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
