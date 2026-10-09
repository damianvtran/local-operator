# One-line installer for Local Operator (Windows, PowerShell 5.1+).
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/damianvtran/local-operator/main/scripts/install.ps1 | iex"
#
# The Windows twin of scripts/install.sh, kept step-for-step identical so the
# README can promise one experience (first-run onboarding audit Q6/U8/D9):
#   1. Getting ready             - install uv if it is missing (astral's installer)
#   2. Installing Local Operator - `uv tool install local-operator` (uv brings Python 3.12+)
#   3. Checking it works         - run `lop --version`
# Each step prints "Step N/3", an ETA and the elapsed total.
#
# ASCII ONLY, and that is a hard constraint rather than a style (review
# round 1, R-6): the header advertises PowerShell 5.1+, and 5.1 decodes a
# BOM-less script in the machine's ANSI code page -- so the tick and the
# middle dots this file used to carry printed as mojibake (`OK` where a
# tick was meant) for anyone running `powershell -File scripts/install.ps1`.
# The documented `irm ... | iex` path was unaffected (the HTTP charset
# decides that decode), which is exactly why it went unnoticed. A BOM
# would fix 5.1 too, but a BOM survives into the string `iex` receives on
# the documented path; ASCII cannot break either route, and a test pins it.
#
# CONSTRAINTS: no admin rights, never touches a system Python or pyenv-win,
# never edits PATH itself (uv's installer and `uv tool update-shell` are the
# documented PATH writers; this script prints the command when it is needed),
# idempotent (`--upgrade`). Overridable: LOCAL_OPERATOR_PACKAGE,
# LOCAL_OPERATOR_PYTHON (default 3.12).

$ErrorActionPreference = 'Stop'

$Package = if ($env:LOCAL_OPERATOR_PACKAGE) { $env:LOCAL_OPERATOR_PACKAGE } else { 'local-operator' }
$PythonVersion = if ($env:LOCAL_OPERATOR_PYTHON) { $env:LOCAL_OPERATOR_PYTHON } else { '3.12' }
$TotalSteps = 3
$Clock = [System.Diagnostics.Stopwatch]::StartNew()

function Get-Elapsed { '{0}s' -f [int]$Clock.Elapsed.TotalSeconds }

function Write-Step([int]$Number, [string]$Label, [string]$Eta) {
    Write-Host ("Step {0}/{1} - {2} " -f $Number, $TotalSteps, $Label) -NoNewline
    Write-Host ("(about {0} - {1} elapsed)" -f $Eta, (Get-Elapsed)) -ForegroundColor DarkGray
}

function Stop-Install([string]$Message) {
    Write-Host "Install failed: $Message" -ForegroundColor Red
    Write-Host ("Elapsed {0}. Re-run the same command to retry; it picks up where it stopped." -f (Get-Elapsed))
    exit 1
}

function Find-Uv {
    $onPath = Get-Command uv -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    foreach ($candidate in @(
            (Join-Path $env:USERPROFILE '.local\bin\uv.exe'),
            (Join-Path $env:USERPROFILE '.cargo\bin\uv.exe'))) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

Write-Host 'Installing Local Operator' -ForegroundColor White

# -- 1. uv ---------------------------------------------------------------------
$Uv = Find-Uv
if ($Uv) {
    Write-Step 1 'Getting ready - uv is already installed' '0 s'
} else {
    Write-Step 1 'Getting ready - installing uv' '10 s'
    try {
        $env:UV_NO_MODIFY_PATH = '1'
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression | Out-Null
    } catch {
        Stop-Install 'could not install uv (https://docs.astral.sh/uv/getting-started/installation/)'
    }
    $Uv = Find-Uv
    if (-not $Uv) { Stop-Install 'uv was installed but cannot be found in %USERPROFILE%\.local\bin' }
}

# -- 2. the CLI ----------------------------------------------------------------
Write-Step 2 ("Installing Local Operator (uv brings Python {0}+)" -f $PythonVersion) '20 s'
& $Uv tool install --upgrade --quiet --python $PythonVersion $Package
if ($LASTEXITCODE -ne 0) { Stop-Install "uv could not install $Package" }

# -- 3. check ------------------------------------------------------------------
Write-Step 3 'Checking it works' '5 s'
$BinDir = (& $Uv tool dir --bin).Trim()
$Lop = Join-Path $BinDir 'lop.exe'
if (-not (Test-Path $Lop)) { Stop-Install "the install finished but $Lop is missing" }
$Version = (& $Lop --version | Select-Object -Last 1)
if ($LASTEXITCODE -ne 0) { Stop-Install 'lop is installed but did not start' }

Write-Host ''
Write-Host ("Local Operator {0} installed in {1}" -f $Version, (Get-Elapsed)) -ForegroundColor Green

$PathParts = ($env:PATH -split ';') | ForEach-Object { $_.TrimEnd('\') }
if ($PathParts -notcontains $BinDir.TrimEnd('\')) {
    Write-Host ''
    Write-Host "$BinDir is not on your PATH yet. Add it with:"
    Write-Host "  & '$Uv' tool update-shell"
    Write-Host 'then open a new terminal.'
}

Write-Host ''
Write-Host 'Next:'
Write-Host '  lop                 start it'
Write-Host '  /login radient      inside it: one browser sign-in (recommended),'
Write-Host '                      or /login for any other provider or an API key'
