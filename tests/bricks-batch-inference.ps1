param(
    [Parameter(Mandatory=$true)][string]$Models,
    [Parameter(Mandatory=$true)][string]$Idle,
    [Parameter(Mandatory=$true)][string]$Walk,
    [string]$Tool=(Join-Path $PSScriptRoot '../build/Release/tide-motion.exe'),
    [string]$Fixture=(Join-Path $PSScriptRoot '../build/Release/motion-tests.exe'),
    [int]$Threads=8
)
# Opt-in: batched MotionBricks connects against one-request runs, run splitting at
# in-batch dependencies, and wall time. Holds TideMotionGpuValidation like the others.
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1'
$work=Join-Path (Split-Path -Parent $PSScriptRoot) ('.research/bricks-batch-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $work | Out-Null
Copy-Item -LiteralPath $Idle -Destination (Join-Path $work 'idle.glb')
Copy-Item -LiteralPath $Walk -Destination (Join-Path $work 'walk.glb')
# Varied independent requests: placements, Poses to Use, fixed and ranged tokens.
$specs=@()
for($k=0;$k -lt 32;++$k) {
    $to=if($k%3 -eq 2){'idle.glb'}else{'walk.glb'}
    $spec=@{action='connect';from='walk.glb';to=$to;offset=@([Math]::Round(0.1*($k%5)-0.2,2),0,(2.6+0.1*($k%7)));targetPoses=(1+$k%4);threads=$Threads;output="b$k.glb"}
    if($k%2){$spec.tokens=@(6,16)}else{$spec.tokens=8+$k%7}
    if($k%4 -eq 0){$spec.joined=$true}
    $specs+=$spec
}
$batched=@{actions=$specs}
# One request per run in the same process: a cheap mirror between connects ends each run.
$singleActions=@();$m=0
foreach($spec in $specs){$c=$spec.Clone();$c.output='s'+$spec.output;$singleActions+=$c;$singleActions+=@{action='mirror';source='idle.glb';output="m$m.glb"};++$m}
$single=@{actions=$singleActions}
# In-batch dependency: the third action reads the first's output, so the run splits.
$chain=@{actions=@(
    @{action='connect';from='walk.glb';to='walk.glb';offset=@(0,0,3);threads=$Threads;output='c0.glb'},
    @{action='connect';from='idle.glb';to='walk.glb';offset=@(0,0,1);threads=$Threads;output='c1.glb'},
    @{action='connect';from='c0.glb';to='idle.glb';offset=@(0,0,6);threads=$Threads;output='c2.glb'},
    @{action='connect';from='walk.glb';to='idle.glb';offset=@(0,0,3);threads=$Threads;output='c3.glb'}
)}
foreach($p in @(@($batched,'batched'),@($single,'single'),@($chain,'chain'))){$p[0]|ConvertTo-Json -Depth 8|Set-Content (Join-Path $work ($p[1]+'.json'))}
$mutex=[Threading.Mutex]::new($false,'TideMotionGpuValidation');$locked=$false
try {
    try {$locked=$mutex.WaitOne(0)} catch [Threading.AbandonedMutexException] {$locked=$true}
    if(!$locked){throw 'TideMotionGpuValidation is busy; retry when other validation finishes'}
    $t=[Diagnostics.Stopwatch]::StartNew()
    & $Tool batch (Join-Path $work 'batched.json') --models $Models *> (Join-Path $work 'batched.log')
    if($LASTEXITCODE){throw "Batched run failed: $work/batched.log"}
    $batchedSeconds=$t.Elapsed.TotalSeconds
    $t=[Diagnostics.Stopwatch]::StartNew()
    & $Tool batch (Join-Path $work 'single.json') --models $Models *> (Join-Path $work 'single.log')
    if($LASTEXITCODE){throw "Single-request run failed: $work/single.log"}
    $singleSeconds=$t.Elapsed.TotalSeconds
    & $Tool batch (Join-Path $work 'chain.json') --models $Models *> (Join-Path $work 'chain.log')
    if($LASTEXITCODE){throw "Chained run failed: $work/chain.log"}
} finally {if($locked){$mutex.ReleaseMutex()};$mutex.Dispose()}
if(@(Select-String -LiteralPath (Join-Path $work 'batched.log') -Pattern 'Loading MotionBricks SOMA').Count -ne 1){throw 'Expected one ONNX session'}
if(@(Select-String -LiteralPath (Join-Path $work 'batched.log') -Pattern 'Generating 32 transitions in one batch').Count -ne 1){throw 'Expected one 32-request run'}
$maxP=0;$maxQ=0
foreach($s in $specs) {
    $line=& $Fixture --compare (Join-Path $work $s.output) (Join-Path $work ('s'+$s.output))
    if($LASTEXITCODE){throw "Batched and single outputs differ in shape: $($s.output)"}
    $f=$line -split ' ';$maxP=[Math]::Max($maxP,[double]$f[3]);$maxQ=[Math]::Max($maxQ,[double]$f[5])
    $a=Get-Content -Raw (Join-Path $work ($s.output+'.json'))|ConvertFrom-Json;$b=Get-Content -Raw (Join-Path $work ('s'+$s.output+'.json'))|ConvertFrom-Json
    if($a.chosenTokens -ne $b.chosenTokens){throw "Batched token choice differs: $($s.output)"}
    if($a.motionbricksBatch.requests -ne 32 -or $b.motionbricksBatch.requests -ne 1){throw 'Batch recipe fields'}
}
Write-Output ("Batched vs single: max position {0:E3} m, max rotation {1:E3} rad over 32 outputs" -f $maxP,$maxQ)
if($maxP -gt 1e-3 -or $maxQ -gt 1e-2){throw 'Batched outputs diverge from single-request outputs'}
$runs=@(Select-String -LiteralPath (Join-Path $work 'chain.log') -Pattern 'one MotionBricks run' | ForEach-Object {$_.Line})
Write-Output ("Chain runs: "+($runs -join '; '))
if($runs.Count -ne 2 -or $runs[0] -notmatch '1-2' -or $runs[1] -notmatch '3-4'){throw 'Dependent connect did not split the run'}
if(@(Select-String -LiteralPath (Join-Path $work 'single.log') -Pattern 'in one batch').Count){throw 'Single-request manifest was batched'}
Write-Output ("Wall time, one process each, 32 connects: batched {0:N2} s; one request per run {1:N2} s (plus 32 mirrors)" -f $batchedSeconds,$singleSeconds)
$inference={param($log) (Select-String -LiteralPath (Join-Path $work $log) -Pattern 'MotionBricks inference: ([0-9.eE+-]+) s' | ForEach-Object {[double]$_.Matches[0].Groups[1].Value} | Measure-Object -Sum).Sum}
Write-Output ("Model time, 32 connects: batched {0:N3} s in one run; one request per run {1:N3} s total" -f (& $inference 'batched.log'),(& $inference 'single.log'))
Write-Output "MotionBricks batching validated: $work"
