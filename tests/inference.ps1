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
    @{ action = 'connect'; from = 'sequence.glb'; to = 'idle.glb'; offset = @(0, 0, 1.7); tokens = 8; output = 'bridge.glb' },
    @{ action = 'generate'; prompts = @(@{ prompt = 'A person walks forward.'; duration = 1.5 }, @{ prompt = 'A person waves with the right hand.'; duration = 1.5 });
       seed = 3; startPose = 'idle.glb'; endPose = @{ file = 'walk.glb'; frame = 0 }; placement = 'free'; output = 'continued.glb' }
)
$manifest = Join-Path $work 'jobs.json'
@{ actions = $actions } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding utf8
$log = Join-Path $work 'batch.log'
& $toolPath batch $manifest --models $modelPath *> $log
if ($LASTEXITCODE -ne 0) { Get-Content -LiteralPath $log; throw 'Real model batch failed' }
if (@(Select-String -LiteralPath $log -Pattern 'Loading Kimodo SOMA').Count -ne 1 -or
    @(Select-String -LiteralPath $log -Pattern 'Loading MotionBricks SOMA').Count -ne 1) { throw 'Backend was loaded more than once' }
foreach ($name in @('idle', 'walk', 'sequence', 'bridge', 'continued')) {
    $clip = Join-Path $work "$name.glb"
    $info = & $toolPath inspect $clip | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or $info.joints -ne 30 -or $info.frames -lt 4) { throw "Invalid exported clip: $name" }
}
# Two prompts from idle's last pose to walk's first: the model lands within a few
# centimetres and correction makes both poses exact.
$recipe = Get-Content -LiteralPath (Join-Path $work 'continued.glb.json') -Raw | ConvertFrom-Json
if ($recipe.frames -ne 90 -or $recipe.prompts.Count -ne 2 -or $recipe.constraints.Count -ne 2) { throw 'Unexpected constrained sequence recipe' }
foreach ($constraint in $recipe.constraints) {
    if ($constraint.modelError.positionMeanMetres -gt 0.06 -or $constraint.error.positionMaxMetres -gt 0.001 -or $constraint.error.rotationMaxDegrees -gt 0.1) {
        throw "Constraint $($constraint.kind) missed: $($constraint | ConvertTo-Json -Depth 4)"
    }
}
if ($Tide) {
    & $toolPath preview (Join-Path $work 'sequence.glb') --tide $Tide --capture (Join-Path $work 'preview.png') --frames 90
    if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath (Join-Path $work 'preview.png'))) { throw 'Tide asset preview failed' }
    & $toolPath preview (Join-Path $work 'continued.glb') --tide $Tide --gif (Join-Path $work 'continued.gif')
    if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath (Join-Path $work 'continued.gif'))) { throw 'Tide GIF preview failed' }
}
Write-Output "Real inference and model-reuse checks passed: $work"
