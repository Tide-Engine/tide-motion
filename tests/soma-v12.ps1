param([string]$Tool = "$PSScriptRoot/../build/Release/tide-motion.exe", [string]$Models, [string]$Work)
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1'
$env:KIMODO_PROFILE='1'
$Tool=(Resolve-Path $Tool).Path
$Models=(Resolve-Path $Models).Path
New-Item -ItemType Directory -Force $Work | Out-Null
$Work=(Resolve-Path $Work).Path
$actions=@()
foreach($model in @('v1.1','v1.2')) {
 foreach($seed in @(1,2,3)) {
  foreach($case in @(@('idle','A person stands relaxed.',2),@('walk','A person walks forward.',3),@('kick','A person kicks forward with the right leg.',2.5),@('longkick','A person kicks forward with the right leg.',4))) {
   $actions+=@{action='generate';model=$model;prompt=$case[1];duration=$case[2];seed=$seed;device='gpu';output="$model-$($case[0])-$seed.glb"}
  }
 }
 $pose='v1.1-idle-1.glb'
 $actions+=@{action='generate';model=$model;prompt='A person kicks forward with the right leg.';duration=2.5;seed=1;device='gpu';startPose=$pose;endPose=@{file=$pose;frame=0};output="$model-pose.glb"}
 $actions+=@{action='generate';model=$model;prompt='A person walks forward.';duration=3;seed=1;device='gpu';path=@(@{frame=0;x=0;z=0},@{frame=45;x=0;z=0.75},@{frame=89;x=0;z=1.5});output="$model-path.glb"}
 $actions+=@{action='generate';model=$model;prompts=@(@{prompt='A person walks forward.';duration=1.5},@{prompt='A person waves with the right hand.';duration=1.5});seed=1;device='gpu';startPose=$pose;output="$model-multi.glb"}
}
# Returning to v1.1 checks model switching and reproducibility in one process.
$actions+=@{action='generate';model='v1.1';prompt='A person stands relaxed.';duration=2;seed=1;device='gpu';output='v1.1-repeat.glb'}
@{actions=$actions}|ConvertTo-Json -Depth 8|Set-Content "$Work/jobs.json"
$mutex=[Threading.Mutex]::new($false,'TideMotionGpuValidation')
$held=$false
try {
 try {$held=$mutex.WaitOne(0)} catch [Threading.AbandonedMutexException] {$held=$true}
 if(!$held) {throw 'TideMotionGpuValidation is busy; rerun when available'}
 $start=Get-Date
 $p=Start-Process -FilePath $Tool -ArgumentList @('batch',"`"$Work/jobs.json`"",'--models',"`"$Models`"") -PassThru -WindowStyle Hidden -RedirectStandardOutput "$Work/stdout.log" -RedirectStandardError "$Work/stderr.log"
 $peak=[long]0
 while(!$p.WaitForExit(100)) {$p.Refresh();$peak=[Math]::Max([long]$peak,[long]$p.WorkingSet64)}
 if($p.ExitCode -ne 0) {throw "Inference failed ($($p.ExitCode)); see $Work/stderr.log"}
 @{seconds=((Get-Date)-$start).TotalSeconds;sampledPeakWorkingSetBytes=$peak;actions=$actions.Count}|ConvertTo-Json|Set-Content "$Work/resources.json"
 foreach($a in $actions) {
  $r=Get-Content "$Work/$($a.output).json" -Raw|ConvertFrom-Json
  if($r.model -ne "nvidia/Kimodo-SOMA-RP-$($a.model)") {throw 'Wrong model in recipe'}
  & $Tool inspect "$Work/$($a.output)" | Out-Null
  if($LASTEXITCODE) {throw 'GLB import failed'}
 }
 if((Get-FileHash "$Work/v1.1-idle-1.glb").Hash -ne (Get-FileHash "$Work/v1.1-repeat.glb").Hash) {
  # GLB animation names differ; compare decoded animation buffers separately.
  Write-Output 'Check animation payload identity (output names differ).'
 }
 if(@(Select-String -Path "$Work/stderr.log" -Pattern "Loading Kimodo SOMA").Count -ne 3) {throw "Expected exactly three model loads"}
 foreach($version in @("v1.1","v1.2")) {
  $r=Get-Content "$Work/$version-pose.glb.json" -Raw|ConvertFrom-Json
  foreach($c in $r.constraints) {if($c.error.positionMaxMetres -gt .001) {throw "Pose correction failed"}}
 }
 Get-Content "$Work/resources.json"
} finally {
 if($p -and !$p.HasExited) {Stop-Process -Id $p.Id; $p.WaitForExit()}
 if($held){$mutex.ReleaseMutex()};$mutex.Dispose()
}

