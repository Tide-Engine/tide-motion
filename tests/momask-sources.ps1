param([switch]$FetchMissing)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot
$destination = Join-Path $root '.research/momask-source'
$lock = Get-Content (Join-Path $root 'notes/momask-source-lock.json') -Raw | ConvertFrom-Json
$seen = @{}
foreach ($record in $lock.files) {
    # Only audited text sources. No archives, checkpoints, dataset or body assets.
    if ($record.path -notmatch '^[A-Za-z0-9_./-]+$' -or $record.path -match '(^|/)\.\.(/|$)' -or
        $record.path -match '^(dataset|example_data|assets|visualization)/' -or
        $record.path -notmatch '(\.py|\.md|\.txt|\.sh|\.yml|LICENSE)$' -or
        $record.sha256 -notmatch '^[a-f0-9]{64}$' -or $seen.ContainsKey($record.path)) {
        throw "Invalid source record: $($record.path)"
    }
    $seen[$record.path] = $true
    if ($record.url -notmatch '^https://raw\.githubusercontent\.com/(EricGuo5513/(momask-codes|HumanML3D)|openai/CLIP)/[a-f0-9]{40}/' -and
        $record.url -notmatch '^https://huggingface\.co/spaces/MeYourHint/MoMask/raw/[a-f0-9]{40}/') {
        throw "Unpinned source URL: $($record.url)"
    }
    $file = Join-Path $destination $record.path
    if (!(Test-Path -LiteralPath $file) -and $FetchMissing) {
        New-Item -ItemType Directory -Force (Split-Path $file) | Out-Null
        & curl.exe --fail --silent --show-error --location --proto '=https' --proto-redir '=https' --max-time 30 $record.url --output $file
        if ($LASTEXITCODE) { throw "Source fetch failed: $($record.path)" }
    }
    if (!(Test-Path -LiteralPath $file) -or
        (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) {
        throw "Missing or changed source: $($record.path)"
    }
}
Write-Output "Verified $($seen.Count) pinned source/documentation files; no upstream code executed or model assets fetched."
