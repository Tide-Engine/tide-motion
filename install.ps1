param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [string]$Root = (Join-Path $env:LOCALAPPDATA 'TideMotion'),
    [switch]$AddToPath
)
$ErrorActionPreference = 'Stop'
$archivePath = (Resolve-Path -LiteralPath $Archive).Path
$checksumPath = "$archivePath.sha256"
if (!(Test-Path -LiteralPath $checksumPath)) { throw "Missing checksum file: $checksumPath" }
$expected = ((Get-Content -LiteralPath $checksumPath -Raw).Trim() -split '\s+')[0]
if ($expected -notmatch '^[0-9a-fA-F]{64}$') { throw 'Malformed archive checksum' }
if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash -ne $expected) { throw 'Archive checksum mismatch' }
$installRoot = [IO.Path]::GetFullPath($Root)
[IO.Directory]::CreateDirectory($installRoot) | Out-Null
$stagePath = Join-Path $installRoot ('.install-' + [guid]::NewGuid().ToString('N'))
$backupPath = Join-Path $installRoot ('.previous-' + [guid]::NewGuid().ToString('N'))
$binPath = Join-Path $installRoot 'bin'
function Assert-InstallChild([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if (!$resolved.StartsWith($installRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes installation: $resolved"
    }
}
function Remove-InstallScratch([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if (!$resolved.StartsWith($installRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing cleanup outside installation: $resolved"
    }
    if (Test-Path -LiteralPath $resolved) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}
try {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        foreach ($entry in $zip.Entries) {
            $entryPath = [IO.Path]::GetFullPath((Join-Path $stagePath $entry.FullName))
            if (!$entryPath.StartsWith($stagePath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
                $entry.FullName.Contains(':') -or (($entry.ExternalAttributes -shr 16) -band 0xF000) -eq 0xA000) {
                throw "Unsafe archive entry: $($entry.FullName)"
            }
        }
    } finally { $zip.Dispose() }
    Expand-Archive -LiteralPath $archivePath -DestinationPath $stagePath
    $packages = @(Get-ChildItem -LiteralPath $stagePath -Directory)
    $payload = $stagePath
    if (!(Test-Path -LiteralPath (Join-Path $payload 'bin/tide-motion.exe')) -and $packages.Count -eq 1) { $payload = $packages[0].FullName }
    $freshBin = Join-Path $payload 'bin'
    Assert-InstallChild $freshBin
    Assert-InstallChild $binPath
    Assert-InstallChild $backupPath
    if (!(Test-Path -LiteralPath (Join-Path $freshBin 'tide-motion.exe')) -or
        !(Test-Path -LiteralPath (Join-Path $freshBin 'onnxruntime.dll')) -or
        !(Test-Path -LiteralPath (Join-Path $payload 'licenses/META-LLAMA-3.txt'))) { throw 'Incomplete Tide Motion package' }
    & (Join-Path $freshBin 'tide-motion.exe') --version
    if ($LASTEXITCODE -ne 0) { throw 'The packaged executable cannot run on this machine' }
    if (Test-Path -LiteralPath $binPath) { Move-Item -LiteralPath $binPath -Destination $backupPath }
    try { Move-Item -LiteralPath $freshBin -Destination $binPath }
    catch {
        if (Test-Path -LiteralPath $backupPath) { Move-Item -LiteralPath $backupPath -Destination $binPath }
        throw
    }
    foreach ($folder in @('licenses', 'examples', 'notes')) {
        Copy-Item -LiteralPath (Join-Path $payload $folder) -Destination $installRoot -Recurse -Force
    }
    Copy-Item -LiteralPath (Join-Path $payload 'README.md') -Destination $installRoot -Force
    Copy-Item -LiteralPath (Join-Path $payload 'LICENSE') -Destination $installRoot -Force
    if ($AddToPath) {
        [string]$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        if (!(($userPath -split ';') -contains $binPath)) {
            [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $binPath).TrimStart(';'), 'User')
        }
    }
    Remove-InstallScratch $backupPath
    Write-Output "Installed Tide Motion in $installRoot"
    Write-Output "Run: `"$binPath/tide-motion.exe`" --help"
} finally { Remove-InstallScratch $stagePath }
