# Developer-only preparation; no Python or download process is used by inference.
param([string]$Destination = (Join-Path (Split-Path $PSScriptRoot) '.research/momask-live'))
$ErrorActionPreference = 'Stop'
$pins = Get-Content (Join-Path (Split-Path $PSScriptRoot) 'notes/momask-private-inputs.json') -Raw | ConvertFrom-Json
foreach ($pin in $pins.files) {
    if ($pin.file -match '(^|/)\.\.(/|$)' -or $pin.file -notmatch '^[A-Za-z0-9_./-]+$' -or $pin.url -notlike 'https://*') { throw 'Invalid private input pin' }
    $file = Join-Path $Destination $pin.file
    if (!(Test-Path -LiteralPath $file)) {
        New-Item -ItemType Directory -Force (Split-Path $file) | Out-Null
        & curl.exe --fail --location --silent --show-error --proto '=https' --proto-redir '=https' --retry 2 --max-time 300 $pin.url --output $file
        if ($LASTEXITCODE) { throw "Download failed: $($pin.file)" }
    }
    if ((Get-Item -LiteralPath $file).Length -ne $pin.size -or (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $pin.sha256) { throw "Input pin mismatch: $($pin.file)" }
}
Write-Output 'Pinned private checkpoint inputs verified. Commercial checkpoint scope remains unresolved; do not redistribute.'
