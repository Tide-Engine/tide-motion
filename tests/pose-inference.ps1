param(
    [Parameter(Mandatory=$true)][string]$Models,
    [Parameter(Mandatory=$true)][string]$Idle,
    [Parameter(Mandatory=$true)][string]$Walk,
    [string]$Tool=(Join-Path $PSScriptRoot '../build/Release/tide-motion.exe'),
    [string]$Fixture=(Join-Path $PSScriptRoot '../build/Release/motion-tests.exe'),
    [string]$Tide
)
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1'
$work=Join-Path (Split-Path -Parent $PSScriptRoot) ('.research/pose-sequence-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $work | Out-Null
Copy-Item -LiteralPath $Idle -Destination (Join-Path $work 'idle.glb')
Copy-Item -LiteralPath $Walk -Destination (Join-Path $work 'walk.glb')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../examples/pose-sequence.json') -Destination (Join-Path $work 'poses.json')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '../examples/pose-sequence-guides.json') -Destination (Join-Path $work 'guides.json')
# Upstream controls: token ranges, Poses to Use, Transition Early By and predicted contacts.
# walk.glb is expected to walk about 2 m forward from the origin (as Kimodo's walk does);
# offset 3 m places the next walk start about 1 m ahead of its end.
@{actions=@(
    @{action='pose-sequence';recipe='poses.json';threads=8;output='authored.glb'},
    @{action='connect';from='idle.glb';to='walk.glb';tokens=8;threads=8;output='compatible.glb'},
    @{action='pose-sequence';recipe='guides.json';threads=8;contactsOut='guides-contacts.json';output='guided.glb'},
    @{action='connect';from='walk.glb';to='walk.glb';offset=@(0,0,3);tokens=@(6,16);targetPoses=4;threads=8;contactsOut='walk-contacts.json';joined=$true;output='walkwalk.glb'},
    @{action='connect';from='walk.glb';to='walk.glb';offset=@(0,0,3);tokens=@(6,16);targetPoses=1;threads=8;output='stop1.glb'},
    @{action='connect';from='walk.glb';to='walk.glb';offset=@(0,0,3);tokens=@(6,16);targetPoses=4;threads=8;output='moving4.glb'}
)} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $work 'jobs.json')
$mutex=[Threading.Mutex]::new($false,'TideMotionGpuValidation')
$locked=$false
try {
    try {$locked=$mutex.WaitOne(0)} catch [Threading.AbandonedMutexException] {$locked=$true}
    if(!$locked){throw 'TideMotionGpuValidation is busy; retry when other validation finishes'}
    & $Tool batch (Join-Path $work 'jobs.json') --models $Models *> (Join-Path $work 'inference.log')
    if($LASTEXITCODE){throw "Native inference failed: $work/inference.log"}
    if(@(Select-String -LiteralPath (Join-Path $work 'inference.log') -Pattern 'Loading MotionBricks SOMA').Count -ne 1){throw 'Expected one shared ONNX session'}
    & $Fixture --verify-sequence (Join-Path $work 'poses.json') (Join-Path $work 'authored.glb')
    if($LASTEXITCODE){throw 'Endpoint/timing verification failed'}
    & $Fixture --verify-sequence (Join-Path $work 'guides.json') (Join-Path $work 'guided.glb')
    if($LASTEXITCODE){throw 'Guided sequence verification failed'}
    foreach($clip in 'walkwalk','stop1','moving4') {
        $recipe=Get-Content -Raw (Join-Path $work "$clip.glb.json") | ConvertFrom-Json
        if($recipe.chosenTokens -lt 6 -or $recipe.chosenTokens -gt 16){throw "Chosen tokens outside the allowed range: $clip"}
        # Raw predictions should meet both seams closely once in MotionBricks' joint frames.
        foreach($seam in $recipe.seamCorrection.rootMetres){if([Math]::Sqrt(($seam|ForEach-Object{$_*$_}|Measure-Object -Sum).Sum) -gt 0.1){throw "Raw root misses a seam by over 10 cm: $clip"}}
    }
    # Poses to Use: one target pose stops there; four keep the target's walking velocity.
    $stop=[double]((& $Fixture --end-speed (Join-Path $work 'stop1.glb')) -split ' ')[-1]
    $moving=[double]((& $Fixture --end-speed (Join-Path $work 'moving4.glb')) -split ' ')[-1]
    Write-Output "Arrival speed: 1 target pose $stop m/s, 4 target poses $moving m/s"
    if($stop -gt 0.3 -or $moving -lt 0.6){throw 'Target pose count did not change the arrival as documented'}
    foreach($plan in 'guides-contacts','walk-contacts') {
        $file=Join-Path $work "$plan.json"
        if(!(Test-Path $file)){throw "Expected a predicted contact plan: $plan"}
        $source=if($plan -eq 'guides-contacts'){'guided.glb'}else{'walkwalk.glb'}
        & $Tool cleanup (Join-Path $work $source) --contacts $file -o (Join-Path $work "$plan-clean.glb")
        if($LASTEXITCODE){throw "Cleanup with predicted contacts failed: $plan"}
    }
    if($Tide) {
        & $Tool preview (Join-Path $work 'authored.glb') --tide $Tide --capture (Join-Path $work 'arrival.png') --frames 51
        if($LASTEXITCODE){throw 'Bounded Tide preview failed'}
    }
} finally {if($locked){$mutex.ReleaseMutex()};$mutex.Dispose()}
Write-Output "Pose authoring inference validated: $work"
