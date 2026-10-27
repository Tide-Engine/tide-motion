param(
    [Parameter(Mandatory = $true)][string]$Models,
    [string]$Tool = (Join-Path $PSScriptRoot '../build/Release/tide-motion.exe'),
    [string]$Tide
)
$ErrorActionPreference = 'Stop'
$toolPath = (Resolve-Path -LiteralPath $Tool).Path
$modelPath = (Resolve-Path -LiteralPath $Models).Path
$repoRoot = Split-Path -Parent $PSScriptRoot
$work = Join-Path $repoRoot ('.research/inference-' + [guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($work) | Out-Null
$actions = @(
    @{ action = 'generate'; prompt = 'A person stands still in a relaxed pose.'; duration = 2; seed = 1; output = 'idle.glb' },
    @{ action = 'generate'; prompt = 'A person walks forward.'; duration = 2; seed = 2; output = 'walk.glb' },
    @{ action = 'connect'; from = 'idle.glb'; to = 'walk.glb'; offset = @(0, 0, 0.4); tokens = 8; joined = $true; output = 'sequence.glb' },
    @{ action = 'connect'; from = 'sequence.glb'; to = 'idle.glb'; offset = @(0, 0, 1.7); tokens = 8; output = 'bridge.glb' }
)
$manifest = Join-Path $work 'jobs.json'
@{ actions = $actions } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding utf8
$log = Join-Path $work 'batch.log'
& $toolPath batch $manifest --models $modelPath *> $log
if ($LASTEXITCODE -ne 0) { Get-Content -LiteralPath $log; throw 'Real model batch failed' }
if (@(Select-String -LiteralPath $log -Pattern 'Loading Kimodo SOMA').Count -ne 1 -or
    @(Select-String -LiteralPath $log -Pattern 'Loading MotionBricks SOMA').Count -ne 1) { throw 'Backend was loaded more than once' }
foreach ($name in @('idle', 'walk', 'sequence', 'bridge')) {
    $clip = Join-Path $work "$name.glb"
    $info = & $toolPath inspect $clip | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or $info.joints -ne 30 -or $info.frames -lt 4) { throw "Invalid exported clip: $name" }
}
if ($Tide) {
    & $toolPath preview (Join-Path $work 'sequence.glb') --tide $Tide --capture (Join-Path $work 'preview.png') --frames 90
    if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath (Join-Path $work 'preview.png'))) { throw 'Tide asset preview failed' }
}
Write-Output "Real inference and model-reuse checks passed: $work"
