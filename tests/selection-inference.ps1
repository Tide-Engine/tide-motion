param(
    [Parameter(Mandatory=$true)][string]$Models,
    [Parameter(Mandatory=$true)][string]$Idle,
    [string]$Tool=(Join-Path $PSScriptRoot '../build/Release/tide-motion.exe'),
    [string]$Fixture=(Join-Path $PSScriptRoot '../build/Release/motion-tests.exe'),
    [ValidateSet('cpu','gpu')][string]$Device='cpu',
    [unsigned]$Threads=4,
    [string]$Tide
)
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1'
$work=Join-Path (Split-Path -Parent $PSScriptRoot) ('.research/selection-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $work | Out-Null
Copy-Item -LiteralPath $Idle -Destination (Join-Path $work 'idle.glb')
# Coordinate CPU measurements too, so this test cannot contaminate another task's timings.
$mutex=[Threading.Mutex]::new($false,'TideMotionGpuValidation');$locked=$false
try {
    try {$locked=$mutex.WaitOne(0)} catch [Threading.AbandonedMutexException] {$locked=$true}
    if(!$locked){throw 'TideMotionGpuValidation is busy; retry when other validation finishes'}
    $base=@{action='generate';model='v1.1';prompt='A person stumbles backwards in two steps';duration=2;startPose='idle.glb';endPose='idle.glb';endAt=@(0,-1,90);device=$Device;threads=$Threads}
    $fit=$base.Clone();$fit.seed=2;$fit.fitPoses=$true;$fit.attempts=2;$fit.output='ranked.glb'
    $counted=$base.Clone();$counted.seed=0;$counted.footsteps=2;$counted.attempts=8;$counted.output='counted.glb'
    $exhausted=$base.Clone();$exhausted.seed=1;$exhausted.footsteps=64;$exhausted.attempts=2;$exhausted.output='never.glb'
    @{actions=@($fit,$counted,$exhausted)} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $work 'jobs.json')
    & $Tool batch (Join-Path $work 'jobs.json') --models $Models *> (Join-Path $work 'inference.log')
    if($LASTEXITCODE -ne 1 -or !(Select-String -Path (Join-Path $work 'inference.log') -Pattern 'No candidate matched --footsteps after 2 attempts')){throw 'Expected bounded exhausted-selection refusal'}
    if((Test-Path (Join-Path $work 'never.glb')) -or (Test-Path (Join-Path $work 'never.glb.json'))){throw 'Exhausted selection wrote output'}
    if(@(Select-String -Path (Join-Path $work 'inference.log') -Pattern 'Loading Kimodo SOMA').Count -ne 1){throw 'Expected one model load'}
    $ranked=Get-Content -Raw (Join-Path $work 'ranked.glb.json') | ConvertFrom-Json
    $best=@($ranked.selection.attempted | Sort-Object poseMeanRotationDegrees,seed)[0]
    if($ranked.seed -ne $best.seed -or $ranked.selection.attempted.Count -ne 2){throw 'Pose ranking selected the wrong seed or attempt count'}
    $counted=Get-Content -Raw (Join-Path $work 'counted.glb.json') | ConvertFrom-Json
    $info=(& $Tool inspect (Join-Path $work 'counted.glb')) | ConvertFrom-Json
    if($info.footfalls.count -ne 2 -or $counted.footfalls.count -ne 2){throw 'Exported footfall count differs'}
    foreach($file in @('ranked.glb','counted.glb')) {
        & $Fixture --verify-placed (Join-Path $work 'idle.glb') (Join-Path $work $file) 0 -1 90
        if($LASTEXITCODE){throw 'Selected export endpoints differ from source/placement'}
    }
    if($Tide) {
        & $Fixture --variant-sheet $Tide (Join-Path $work 'selected.png') (Join-Path $work 'ranked.glb') (Join-Path $work 'counted.glb')
        if($LASTEXITCODE){throw 'Bounded variant sheet failed'}
    }
} finally {if($locked){$mutex.ReleaseMutex()};$mutex.Dispose()}
Write-Output "Counted/pose-ranked native selection validated: $work"
