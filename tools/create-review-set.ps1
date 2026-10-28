param(
    [Parameter(Mandatory=$true)][string]$Tool,
    [Parameter(Mandatory=$true)][string]$FixtureTool,
    [Parameter(Mandatory=$true)][string]$AssetsRoot,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$ExistingGeneratedClip = ''
)
$ErrorActionPreference = 'Stop'
$env:TIDE_HIDDEN = '1'
$toolPath = (Resolve-Path -LiteralPath $Tool).Path
$fixturePath = (Resolve-Path -LiteralPath $FixtureTool).Path
$mapping = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../examples/synty-polygon.json')).Path
$assetRootPath = (Resolve-Path -LiteralPath $AssetsRoot).Path
$assetBase = Join-Path $assetRootPath 'ANIMATION_Sword_Combat_SourceFiles_v5/SourceFiles/Animations/Polygon'
$sources = @(
    @('Idle/Base/A_Idle_Base_Sword.fbx', 'idle'),
    @('Block/A_Parry_F_Sword.fbx', 'parry'),
    @('Dodge/A_DodgeRoll_F_RootMotion_Sword.fbx', 'roll')
)
foreach ($item in $sources) { if (-not (Test-Path -LiteralPath (Join-Path $assetBase $item[0]) -PathType Leaf)) { throw ('Missing source: ' + $item[0]) } }
if ($ExistingGeneratedClip) { $ExistingGeneratedClip = (Resolve-Path -LiteralPath $ExistingGeneratedClip).Path }
if (Test-Path -LiteralPath $Output) { throw 'Choose a new output directory; existing review/feedback is preserved.' }
$outputPath = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Path $outputPath | Out-Null
foreach ($item in $sources) {
    & $toolPath retarget (Join-Path $assetBase $item[0]) --to-soma --map $mapping -o (Join-Path $outputPath ($item[1] + '.glb'))
    if ($LASTEXITCODE -ne 0) { throw ('Retarget failed: ' + $item[1]) }
}
foreach ($name in @('idle','parry')) {
    & $fixturePath (Join-Path $outputPath ($name + '.glb')) (Join-Path $outputPath ('diagnostic-' + $name))
    if ($LASTEXITCODE -ne 0) { throw ('Diagnostic fixture failed: ' + $name) }
}
$guides = @{
    schema = 'tide-motion-pose-sequence-1'
    start = @{file='idle.glb';frame=27;position=@(0,.95,0);yaw=0;hold=$true}
    targets = @(
        @{file='diagnostic-parry/guide-pose.glb';frame=0;position=@(0,.95,0);yaw=0;tokens=8;hold=$true},
        @{file='idle.glb';frame=27;position=@(0,.95,0);yaw=0;tokens=8;hold=$true}
    )
}
$guides | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $outputPath 'guides.json') -Encoding utf8
& $toolPath pose-sequence (Join-Path $outputPath 'guides.json') --threads 8 -o (Join-Path $outputPath 'generated-guides.glb')
if ($LASTEXITCODE -ne 0) { throw 'Native guided inference failed; completed diagnostic assets remain available.' }
$cases = [Collections.Generic.List[object]]::new()
function Add-Case($Id,$File,$Context,$Map='',$Settings=$null) {
    $entry = @{id=$Id;label=('Case {0:D2}' -f ($cases.Count+1));file=$File;context=$Context}
    if ($Map) { $entry.map=$Map }; if ($null -ne $Settings) { $entry.settings=$Settings }; $cases.Add($entry)
}
foreach ($item in $sources) { Add-Case ('library-'+$item[1]) (Join-Path $assetBase $item[0]) 'Unmodified purchased-library source, explicitly retargeted for SOMA proxy review. Original skin and original poses are not certified by this presentation.' $mapping }
if ($ExistingGeneratedClip) { Add-Case 'existing-generation' $ExistingGeneratedClip 'Existing generated animation; inspect its adjacent recipe for original settings.' }
Add-Case 'generated-guides' 'generated-guides.glb' 'Fresh native MotionBricks inference from library idle and extracted parry guide, then back to idle. Two held 8-token targets. See guides.json and the GLB recipe.'
Add-Case 'lowered-reference' 'diagnostic-idle/lowered.glb' 'Controlled diagnostic mutation of retargeted idle: root lowered .15 m. Explicit test edit, not model output.'
Add-Case 'wrist-pulse' 'diagnostic-idle/rotation-step.glb' 'Controlled diagnostic mutation of retargeted idle: +70 degree local wrist rotation for three samples. Explicit test edit, not model output.'
$props = Get-Content -Raw -LiteralPath (Join-Path $outputPath 'diagnostic-parry/prop-settings.json') | ConvertFrom-Json
Add-Case 'prop-probe' 'diagnostic-parry/guide-pose.glb' 'Extracted parry pose with an intentionally misaligned capsule from hand into torso. Not the original sword mesh or calibrated equipment socket.' '' $props
Add-Case 'guide-pose' 'diagnostic-parry/guide-pose.glb' 'Extracted middle parry pose used as a guide to native inference. No test blade.'
@{version=1;title='Tide Motion - human calibration set';cases=$cases.ToArray()} | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $outputPath 'review-set.json') -Encoding utf8
& $toolPath review (Join-Path $outputPath 'review-set.json') -o (Join-Path $outputPath 'calibration.html')
if ($LASTEXITCODE -ne 0) { throw 'Review assembly failed' }
Write-Output ('Open ' + (Join-Path $outputPath 'calibration.html') + '. Export feedback JSON before closing.')
