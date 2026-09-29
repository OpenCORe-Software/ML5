# CORe ML5 installer for Windows.
# - Ensures the ML5 bin directory exists and ml5/ml5d are present
# - Adds the bin directory to the user PATH if not already there
# Usage:
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -ExecutionPolicy Bypass -File install.ps1 -SourceDir .\target\release

param(
    [string]$SourceDir = "",
    [string]$InstallDir = "$env:LOCALAPPDATA\Programs\ML5\bin"
)

$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host "[ml5] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[ml5] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "[ml5] $msg" -ForegroundColor Yellow }

# --- Resolve binaries ---------------------------------------------------------
$candidates = @()
if ($SourceDir) { $candidates += $SourceDir }
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$candidates += @(
    (Join-Path $scriptRoot "target\release"),
    (Join-Path $scriptRoot "target\debug"),
    $scriptRoot
)

$ml5 = $null; $ml5d = $null
foreach ($dir in $candidates) {
    $m = Join-Path $dir "ml5.exe"
    $d = Join-Path $dir "ml5d.exe"
    if ((Test-Path $m) -and (Test-Path $d)) { $ml5 = $m; $ml5d = $d; break }
}

if (-not $ml5) {
    Write-Warn "ml5.exe / ml5d.exe not found in the usual places."
    Write-Warn "Build first:  cargo build --release   (then re-run this script)"
    Write-Warn "Or pass -SourceDir <folder-containing-the-exes>."
    exit 1
}
Write-Info "Found binaries in $(Split-Path $ml5)"

# --- Install ------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
foreach ($exe in @("ml5.exe", "ml5d.exe")) {
    $target = Join-Path $InstallDir $exe
    $old = Join-Path $InstallDir ("$exe.old-$stamp")
    if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old -Force -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $target) {
        try {
            Move-Item -LiteralPath $target -Destination $old -Force -ErrorAction Stop
            Write-Info "Moved running $exe aside as $(Split-Path $old -Leaf)"
        } catch {
            Write-Warn "Cannot move $exe because it is locked. Close running ML5 windows and retry."
            throw
        }
    }
}

try {
    Copy-Item $ml5  (Join-Path $InstallDir "ml5.exe")  -Force -ErrorAction Stop
    Copy-Item $ml5d (Join-Path $InstallDir "ml5d.exe") -Force -ErrorAction Stop
} catch {
    Write-Warn "Install failed after replacing binaries; old copies are kept beside ml5.exe/ml5d.exe."
    throw
}
Write-Ok "Installed to $InstallDir"

# --- PATH ---------------------------------------------------------------------
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$entries = @()
if ($userPath) { $entries = $userPath -split ";" | Where-Object { $_ -ne "" } }

$already = $entries | Where-Object { $_.TrimEnd("\") -ieq $InstallDir.TrimEnd("\") }
if ($already) {
    Write-Info "PATH already contains $InstallDir"
} else {
    $newPath = ($entries + $InstallDir) -join ";"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    $env:Path = "$env:Path;$InstallDir"
    Write-Ok "Added $InstallDir to user PATH (open a new terminal to pick it up everywhere)"
}

Write-Ok "Done. Try:  ml5 status"
