param([Parameter(Mandatory = $true)][string]$Archive)
$ErrorActionPreference = 'Stop'
$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$repoRoot = Split-Path -Parent $PSScriptRoot
$work = Join-Path $repoRoot ('.research/installer-test-' + [guid]::NewGuid().ToString('N'))
$root = Join-Path $work "with spaces and 'quotes'"
[IO.Directory]::CreateDirectory((Join-Path $root 'models')) | Out-Null
Set-Content -LiteralPath (Join-Path $root 'models/keep.txt') -Value 'keep'
& (Join-Path $repoRoot 'install.ps1') -Archive $archivePath -Root $root
& (Join-Path $repoRoot 'install.ps1') -Archive $archivePath -Root $root
if ((Get-Content -LiteralPath (Join-Path $root 'models/keep.txt')) -ne 'keep') { throw 'Reinstall changed model data' }
$before = (Get-FileHash -LiteralPath (Join-Path $root 'bin/tide-motion.exe')).Hash
Add-Type -AssemblyName System.IO.Compression.FileSystem
$unsafe = Join-Path $work 'unsafe.zip'
$zip = [IO.Compression.ZipFile]::Open($unsafe, [IO.Compression.ZipArchiveMode]::Create)
try { $zip.CreateEntry('../escape.txt') | Out-Null } finally { $zip.Dispose() }
$hash = (Get-FileHash -LiteralPath $unsafe).Hash
Set-Content -LiteralPath "$unsafe.sha256" -Value $hash -Encoding ascii
$rejected = $false
try { & (Join-Path $repoRoot 'install.ps1') -Archive $unsafe -Root $root }
catch { if ($_.Exception.Message -notmatch 'Unsafe archive entry') { throw }; $rejected = $true }
if (!$rejected) { throw 'Installer accepted path traversal' }
Set-Content -LiteralPath "$unsafe.sha256" -Value ('0' * 64) -Encoding ascii
$rejected = $false
try { & (Join-Path $repoRoot 'install.ps1') -Archive $unsafe -Root $root }
catch { if ($_.Exception.Message -notmatch 'checksum mismatch') { throw }; $rejected = $true }
if (!$rejected) { throw 'Installer accepted a bad checksum' }
if ((Get-FileHash -LiteralPath (Join-Path $root 'bin/tide-motion.exe')).Hash -ne $before) { throw 'Failed install changed the executable' }
$savedPath = $env:PATH
try {
    $env:PATH = "$env:SystemRoot/System32;$env:SystemRoot"
    & (Join-Path $root 'bin/tide-motion.exe') --version
    if ($LASTEXITCODE -ne 0) { throw 'Package depends on development PATH' }
} finally { $env:PATH = $savedPath }
# Retain this small isolated installation for diagnosing a failure; no user
# installation, user PATH or model cache was changed by this test.
Write-Output "Installer checks passed: $root"
