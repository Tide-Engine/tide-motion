param(
    [Parameter(Mandatory=$true)][string]$Models,
    [Parameter(Mandatory=$true)][string]$Idle,
    [string]$Tool=(Join-Path $PSScriptRoot '../build/Release/tide-motion.exe'),
    [string]$Fixture=(Join-Path $PSScriptRoot '../build/Release/motion-tests.exe'),
    [string]$Baseline,
    [string]$Tide
)
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1'
$work=Join-Path (Split-Path -Parent $PSScriptRoot) ('.research/placement-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $work | Out-Null
Copy-Item -LiteralPath $Idle -Destination (Join-Path $work 'idle.glb')
$actions=@()
foreach($seed in @(1,3)) {
    foreach($entry in @(@('stumble','A person stumbles backwards'),@('walk','A person walks backwards'))) {
        $actions+=@{action='generate';prompt=$entry[1];duration=2;seed=$seed;steps=50;startPose='idle.glb';endPose='idle.glb';placement='free';device='gpu';output=($entry[0]+$seed+'.glb')}
    }
}
$mutex=[Threading.Mutex]::new($false,'TideMotionGpuValidation');$locked=$false
try {
    try {$locked=$mutex.WaitOne(0)} catch [Threading.AbandonedMutexException] {$locked=$true}
    if(!$locked){throw 'TideMotionGpuValidation is busy; retry when validation finishes'}
    if($Baseline) {
        $old=@($actions | ForEach-Object {$a=$_.Clone();$a.output='old-'+$a.output;$a})
        @{actions=$old} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $work 'baseline.json')
        & $Baseline batch (Join-Path $work 'baseline.json') --models $Models *> (Join-Path $work 'baseline.log')
        if($LASTEXITCODE){throw 'Baseline inference failed'}
    }
    $actions+=@{action='generate';prompt='A person stumbles backwards';duration=2;seed=1;steps=50;startPose='idle.glb';endPose='idle.glb';endAt=@(0,-1,90);device='gpu';output='placed.glb'}
    if($Tide){$actions+=@{action='generate';prompt='A person stumbles backwards';duration=2;seed=5;steps=50;startPose='idle.glb';endPose='idle.glb';endAt=@(0,-1);device='gpu';variants=2;tide=$Tide;output='sheet.glb'}}
    @{actions=$actions} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $work 'jobs.json')
    & $Tool batch (Join-Path $work 'jobs.json') --models $Models *> (Join-Path $work 'inference.log')
    if($LASTEXITCODE){throw 'Placement inference failed'}
    & $Fixture --verify-placed (Join-Path $work 'idle.glb') (Join-Path $work 'placed.glb') 0 -1 90
    if($LASTEXITCODE){throw 'Exported endpoint verification failed'}
    foreach($action in $actions) {
        if($action.ContainsKey('variants')){continue}
        $recipe=Get-Content -Raw (Join-Path $work ($action.output+'.json')) | ConvertFrom-Json
        $expected=if($action.placement -eq 'free'){2}else{1}
        if($recipe.freePlacementPasses -ne $expected){throw 'Wrong learned pass count'}
        foreach($constraint in $recipe.constraints) {
            if($constraint.kind -eq 'path'){continue}
            if($constraint.error.positionMaxMetres -gt 0.0002 -or $constraint.error.rotationMaxDegrees -gt 0.1){throw 'Corrected endpoint is inexact'}
        }
    }
    if($Tide -and !(Test-Path (Join-Path $work 'sheet.variants.png'))){throw 'Missing contact sheet'}
    $rows=@(Get-ChildItem $work -Filter '*.glb' | Where-Object Name -ne 'idle.glb' | ForEach-Object {
        $info=(& $Tool inspect $_.FullName) | ConvertFrom-Json
        if($LASTEXITCODE){throw 'Cannot inspect generated output'}
        [pscustomobject]@{clip=$_.Name;displacement=$info.rootTravel.displacementMetres;path=$info.rootTravel.pathMetres;peakSpeed=$info.rootTravel.peakSpeedMetresPerSecond}
    })
    $rows | Export-Csv -NoTypeInformation (Join-Path $work 'travel.csv')
    $rows | Format-Table
} finally {if($locked){$mutex.ReleaseMutex()};$mutex.Dispose()}
Write-Output "Placement inference validated: $work"
