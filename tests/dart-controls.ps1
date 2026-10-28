param([Parameter(Mandatory=$true)][string]$Tool,[Parameter(Mandatory=$true)][string]$Models,[ValidateSet('cpu','gpu','auto')][string]$Device='gpu',[string]$Python)
$ErrorActionPreference='Stop';$env:TIDE_HIDDEN='1'
$Tool=(Resolve-Path -LiteralPath $Tool).Path;$Models=(Resolve-Path -LiteralPath $Models).Path
$work=Join-Path (Split-Path -Parent $PSScriptRoot) ('.research/dart-controls-'+[guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($work)|Out-Null;$env:TEMP=$work;$env:TMP=$work
$actions=[Collections.Generic.List[object]]::new()
function Add-Plan($name,$plan){$plan|ConvertTo-Json -Depth 20|Set-Content -LiteralPath "$work/$name.json";$actions.Add(@{action='dart';plan="$name.json";output="$name.glb";threads=4;device=$Device})}
Add-Plan 'target' @{version=1;seed=3;segments=@(@{text='walk forward';chunks=6})}
Add-Plan 'free' @{version=1;seed=7;segments=@(@{text='wave both hands';chunks=12;guidance=4},@{text='squat down';chunks=12},@{text='jump up';chunks=8})}
Add-Plan 'history' @{version=1;seed=9;history='free.glb.json';segments=@(@{text='walk backwards';chunks=8})}
Add-Plan 'source' @{version=1;seed=9;source='free.glb';sourceFrame=97;segments=@(@{text='turn left and walk';chunks=8})}
Add-Plan 'repeat' @{version=1;seed=4;source='target.glb';sourceFrame=0;seedMode='repeat';sampler='ddpm';timesteps=@(0,3,6,9);guidance=2;fixFloor=$true;groundHeight=.15;segments=@(@{text='';chunks=4})}
Add-Plan 'mean' @{version=1;seed=4;policyMode='mean';segments=@(@{text='walk';chunks=64;target=@(0,2)})}
Add-Plan 'untrained-policy' @{version=1;seed=5;segments=@(@{text='walk backwards';chunks=16;target=@(0,-1)})}
foreach($seed in @(5,8)) {Add-Plan "endpose-$seed" @{version=1;seed=$seed;history='target.glb.json';historyState='first';segments=@(@{text='walk forward';chunks=6});optimization=@{target='target.glb';iterations=120;learningRate=.05;checkGradient=$true}}}
# Self-authored analytic half-space SDF, x-fastest. Positive x is free space.
@{version=1;dimensions=@(2,2,2);origin=@(-3,-3,-3);spacing=@(6,6,6);values=@(-3,3,-3,3,-3,3,-3,3)}|ConvertTo-Json|Set-Content -LiteralPath "$work/plane.json"
Add-Plan 'scene' @{version=1;seed=2;history='target.glb.json';historyState='first';fixFloor=$true;groundHeight=.15;sampler='ddpm';timesteps=@(0,3,6,9);segments=@(@{text='step sideways';chunks=3});optimization=@{iterations=3;learningRate=.03;checkGradient=$true;jerkWeight=.03;floorWeight=.02;scene=@{file='plane.json';margin=.1;weight=5};constraints=@(@{frame=25;joint='LeftHand';position=@(.5,1.5,0);weight=.1})}}
Add-Plan 'predicted' @{version=1;seed=2;history='target.glb.json';historyState='first';eta=.3;timesteps=@(0,4,9);segments=@(@{text='walk';chunks=3;target=@(.2,.3)});optimization=@{jointSpace='predicted';iterations=3;checkGradient=$true}}
@{actions=$actions.ToArray()}|ConvertTo-Json -Depth 20|Set-Content -LiteralPath "$work/batch.json"
$mx=[Threading.Mutex]::new($false,'TideMotionGpuValidation');$held=$false
try {
 try{$held=$mx.WaitOne(0)}catch [Threading.AbandonedMutexException] {$held=$true}
 if(!$held){throw 'TideMotionGpuValidation is occupied; retry later'}
 $info=[Diagnostics.ProcessStartInfo]::new($Tool);$info.UseShellExecute=$false;$info.CreateNoWindow=$true;$info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
 foreach($arg in @('batch',"$work/batch.json",'--models',$Models)){$info.ArgumentList.Add($arg)}
 $process=[Diagnostics.Process]::new();$process.StartInfo=$info;[void]$process.Start();$stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync();$peak=0L
 while(!$process.WaitForExit(100)){$process.Refresh();$peak=[Math]::Max($peak,$process.WorkingSet64)}
 $log=$stdout.Result+"`n"+$stderr.Result;[IO.File]::WriteAllText("$work/run.log",$log)
 if($process.ExitCode){throw "DART controls failed: $work`n$log"}
 if(([regex]::Matches($log,'Loading DART native')).Count -ne 1){throw 'DART sessions were not reused'}
 $summary=@()
 foreach($action in $actions){$r=Get-Content -Raw -LiteralPath (Join-Path $work ($action.output+'.json'))|ConvertFrom-Json
  & $Tool inspect (Join-Path $work $action.output)|Out-Null;if($LASTEXITCODE){throw 'Generated control asset invalid'}
  if($r.metrics.optimization){$o=$r.metrics.optimization;if($o.finalObjective.total -gt $o.initialObjective.total+1e-6){throw 'Optimizer retained a worse candidate'}
   if($action.output -like 'endpose-*' -and ($o.constraints|Measure-Object -Property exportedErrorMetres -Maximum).Maximum -gt .05){throw 'Reachable pose target exceeds 5 cm'}
   if($o.objectiveSpace -like 'canonical*' -and $o.somaForwardMaximumErrorMetres -gt .0002){throw 'Canonical FK parity failed'}
   $summary+=@{asset=$action.output;initial=$o.initialObjective.total;final=$o.finalObjective.total;maxConstraintError=($o.constraints|Measure-Object -Property exportedErrorMetres -Maximum).Maximum;gradientChecks=$o.gradientChecks;seconds=$r.metrics.seconds;gpuPeakLocalBytes=$r.metrics.gpuPeakLocalBytes}
  }
 }
 @{workingSetBytes=$peak;optimization=$summary}|ConvertTo-Json -Depth 20|Set-Content -LiteralPath "$work/summary.json"
 if($Python){& $Python "$PSScriptRoot/dart_controls_metrics.py" $work > "$work/independent.json";if($LASTEXITCODE){throw 'Independent control FK validation failed'}}
 Write-Output "DART controls passed: $work";Get-Content "$work/summary.json"
}finally{if($held){$mx.ReleaseMutex()};$mx.Dispose()}
