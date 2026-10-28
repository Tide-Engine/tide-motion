param([Parameter(Mandatory=$true)][string]$Checkpoint,
      [string]$Probe = "$PSScriptRoot/../../../build-ardy/Release/ardy-native-probe.exe")
$ErrorActionPreference = 'Stop'
$env:TIDE_HIDDEN = '1'
# Download separately from the pinned URL in notes/ardy-feasibility.md.
# No weights are ever installed or included in a Tide runtime package.
$expected = '532ead2c40f27733556c099e8d77e8f227e4047b5a88d556b995d0b9c012dec7'
if ((Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) {
    throw 'Checkpoint SHA-256 differs from the pinned commercial SOMA KimodoRep denoiser'
}
& $Probe $Checkpoint
if ($LASTEXITCODE -ne 0) { throw 'Native projection failed' }
$scratch = Join-Path ([IO.Path]::GetTempPath()) ('ardy-probe-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $scratch | Out-Null
try {
    $bad = Join-Path $scratch 'bad.safetensors'
    [IO.File]::WriteAllBytes($bad, [BitConverter]::GetBytes([uint64]::MaxValue))
    & $Probe $bad
    if ($LASTEXITCODE -eq 0) { throw 'Accepted oversized header' }
    $inputFile = [IO.File]::OpenRead((Resolve-Path -LiteralPath $Checkpoint))
    try {
        $prefix = [byte[]]::new(65536)
        $n = $inputFile.Read($prefix, 0, $prefix.Length)
        if ($n -ne $prefix.Length) { throw 'Could not read truncation fixture' }
        [IO.File]::WriteAllBytes($bad, $prefix)
    } finally { $inputFile.Dispose() }
    & $Probe $bad
    if ($LASTEXITCODE -eq 0) { throw 'Accepted truncated checkpoint' }
    & $Probe
    if ($LASTEXITCODE -eq 0) { throw 'Accepted missing argument' }
} finally {
    # The unique temp directory is the sole deletion target.
    Remove-Item -LiteralPath $bad -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $scratch
}
Write-Output 'PASS: pinned native projection; oversized header, truncation and usage refusals'
