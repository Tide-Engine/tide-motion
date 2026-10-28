# Opt-in real-inference validation of the Kimodo capability work: SOMA-77 output, dense
# root paths, the native Kimodo MotionCorrection port against Tide's correction, the
# runtime options and mapped-humanoid constraint poses. Needs installed models and a
# self-generated SOMA idle clip; optional mapped character/clip for --pose-map. Holds
# TideMotionGpuValidation for every inference. Nothing it writes belongs in Git.
param(
    [Parameter(Mandatory)][string]$Tool,
    [Parameter(Mandatory)][string]$Models,
    [Parameter(Mandatory)][string]$Idle,
    [Parameter(Mandatory)][string]$Work,
    [string]$Tide,
    [string]$Baseline,
    [string]$PoseSource,
    [string]$PoseMap,
    [string]$Character,
    [int]$WaitMinutes=120,
    [ValidateSet('full','correction')][string]$Set='full'
)
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1'
if(Test-Path $Work){throw "Work directory exists: $Work"}
New-Item -ItemType Directory $Work | Out-Null
Copy-Item $Idle (Join-Path $Work 'idle.glb')
$actions=[System.Collections.Generic.List[object]]::new()
function gen($name,$prompt,$duration,$extra){$a=[ordered]@{action='generate';prompt=$prompt;duration=$duration;steps=50;output="$name.glb"};foreach($k in $extra.Keys){$a[$k]=$extra[$k]};$actions.Add($a)}
$modes='none','tide','kimodo'
foreach($seed in 1,2,3){
    foreach($m in 'none','kimodo'){gen "walk-$m-$seed" 'A person walks forward' 3 @{seed=$seed;correction=$m}}
    foreach($m in $modes){gen "kick-$m-$seed" 'A person kicks forward with the right leg' 2.5 @{seed=$seed;correction=$m;startPose='idle.glb';endPose=[ordered]@{file='idle.glb';frame=0}}}
    foreach($m in $modes){gen "path-$m-$seed" 'A person walks forward' 3 @{seed=$seed;correction=$m;path=@([ordered]@{frame=89;x=1.0;z=1.6})}}
}
gen 'wave' 'A person raises the right hand above the head and waves' 3 @{seed=1;correction='none'}
foreach($seed in 1,2,3){foreach($m in $modes){gen "reach-$m-$seed" 'A person stands and raises the right hand' 3 @{seed=$seed;correction=$m;startPose='idle.glb';effectors=@([ordered]@{effectors='right-hand';frame=60;file='wave.glb';sourceFrame=45})}}}
if($Set -eq 'full'){
# Dense versus sparse root paths over the same waypoints (an S-curve); model adherence
# without correction, and with the Kimodo correction's 4 cm root margin.
$waypoints=@([ordered]@{frame=40;x=0.6;z=0.9},[ordered]@{frame=80;x=1.5;z=1.2},[ordered]@{frame=119;x=1.8;z=2.2})
foreach($seed in 1,2,3){
    gen "sparse-$seed" 'A person walks' 4 @{seed=$seed;correction='none';path=$waypoints}
    gen "dense-$seed" 'A person walks' 4 @{seed=$seed;correction='none';path=$waypoints;densePath=$true}
    gen "dense-kimodo-$seed" 'A person walks' 4 @{seed=$seed;correction='kimodo';path=$waypoints;densePath=$true}
}
# SOMA-77 output (same seed as walk-none-1) and every downstream path from it.
gen 'walk77' 'A person walks forward' 3 @{seed=1;correction='none';rig='soma77'}
$actions.Add([ordered]@{action='connect';from='walk77.glb';to='idle.glb';tokens=8;joined=$true;output='walk77-to-idle.glb'})
$actions.Add([ordered]@{action='mirror';source='walk77.glb';output='walk77-mirror.glb'})
gen 'kick77' 'A person kicks forward with the right leg' 2.5 @{seed=1;startPose='walk77.glb';rig='soma77';correction='kimodo'}
if($PoseSource -and $PoseMap){
    Copy-Item $PoseMap (Join-Path $Work 'pose-map.json')
    gen 'mapped-start' 'A person walks forward' 2 @{seed=1;correction='tide';startPose=$PoseSource;poseMap='pose-map.json'}
    if($Character){$actions.Add([ordered]@{action='retarget';source='walk77.glb';target=$Character;map='pose-map.json';output='walk77-character.glb'})}
}
}
$jobs=Join-Path $Work 'jobs.json'
@{actions=$actions} | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $jobs
$mutex=[Threading.Mutex]::new($false,'TideMotionGpuValidation');$locked=$false
try {
    try {$locked=$mutex.WaitOne([TimeSpan]::FromMinutes($WaitMinutes))} catch [Threading.AbandonedMutexException] {$locked=$true}
    if(!$locked){throw 'TideMotionGpuValidation stayed busy'}
    Write-Output "Holding TideMotionGpuValidation from $(Get-Date -Format o)"
    $watch=[Diagnostics.Stopwatch]::StartNew()
    & $Tool batch $jobs --models $Models *> (Join-Path $Work 'batch.log')
    if($LASTEXITCODE){throw "Batch failed; see $Work/batch.log"}
    Write-Output "Batch: $($actions.Count) actions in $([math]::Round($watch.Elapsed.TotalSeconds,1)) s"
    if($Set -eq 'full'){
    # Runtime options: the same CPU generation with 2 and 8 threads, and the default
    # GPU output against a baseline executable that still used environment variables.
    foreach($t in 2,8){
        $w=[Diagnostics.Stopwatch]::StartNew()
        & $Tool generate 'A person walks forward' --duration 1 --steps 10 --seed 4 --device cpu --threads $t -o (Join-Path $Work "cpu-$t.glb") --models $Models *> (Join-Path $Work "cpu-$t.log")
        if($LASTEXITCODE){throw "CPU $t failed"}
        Write-Output "CPU threads ${t}: $([math]::Round($w.Elapsed.TotalSeconds,1)) s"
    }
    if($Baseline){
        & $Baseline generate 'A person walks forward' --duration 3 --seed 1 --no-correction -o (Join-Path $Work 'baseline-walk.glb') --models $Models *> (Join-Path $Work 'baseline.log')
        if($LASTEXITCODE){throw 'Baseline failed'}
        & $Tool generate 'A person walks forward' --duration 3 --seed 1 --no-correction -o (Join-Path $Work 'current-walk.glb') --models $Models *> (Join-Path $Work 'current.log')
        if($LASTEXITCODE){throw 'Current standalone generation failed'}
    }
    if($Tide){
        foreach($c in 'walk77','kick77','walk77-to-idle','dense-1','reach-kimodo-1'){
            & $Tool preview (Join-Path $Work "$c.glb") --tide $Tide --capture (Join-Path $Work "$c.png") --frames 45 *> (Join-Path $Work "$c-preview.log")
            if($LASTEXITCODE){throw "Preview $c failed"}
        }
        if($Character -and (Test-Path (Join-Path $Work 'walk77-character.glb'))){
            & $Tool preview (Join-Path $Work 'walk77.glb') --character $Character --map (Join-Path $Work 'pose-map.json') --tide $Tide --capture (Join-Path $Work 'walk77-character.png') --frames 45 *> (Join-Path $Work 'character-preview.log')
            if($LASTEXITCODE){throw 'Character preview failed'}
        }
    }
    }
} finally {if($locked){$mutex.ReleaseMutex()};$mutex.Dispose()}
Write-Output "Capability inference finished: $Work"
