param([Parameter(Mandatory=$true)][string]$Tool,
      [Parameter(Mandatory=$true)][string]$Models,
      [Parameter(Mandatory=$true)][string]$History,
      [Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1'
$Tool=(Resolve-Path -LiteralPath $Tool).Path
$Models=(Resolve-Path -LiteralPath $Models).Path
$History=(Resolve-Path -LiteralPath $History).Path
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$actions=@(
 @{action='generate';backend='ardy';prompt='A person walks forward at a relaxed pace.';duration=6.1;steps=10;threads=8;correct=$false;output='cold.glb'},
 @{action='generate';backend='ardy';prompt='A person walks forward at a relaxed pace.';duration=6.1;steps=10;threads=8;correct=$false;output='warm.glb'},
 @{action='generate';backend='ardy';prompts=@(@{prompt='A person walks forward at a relaxed pace.';duration=2},@{prompt='A person waves with their right hand.';duration=2.1});history=$History;historyFrames=64;steps=10;threads=8;correct=$false;path=@(@{frame=100;x=1.5;z=1.5});output='history-prompts.glb'},
 @{action='generate';backend='ardy';prompt='A person walks forward at a relaxed pace.';duration=6.1;steps=10;threads=8;correct=$false;keyframes=@(@{frame=90;file=$History;sourceFrame=90});effectors=@(@{frame=30;file=$History;sourceFrame=30;effectors='right-hand'});path=@(@{frame=150;x=2;z=1});output='constrained.glb'},
 @{action='generate';backend='ardy';prompt='A person walks forward at a relaxed pace.';duration=60;steps=4;threads=8;correct=$false;output='long.glb'}
)
$manifest=Join-Path $OutputDirectory 'jobs.json'
@{actions=$actions}|ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifest -Encoding utf8
$mx=[Threading.Mutex]::new($false,'TideMotionGpuValidation');$held=$false
try {
    try {$held=$mx.WaitOne(0)}catch [Threading.AbandonedMutexException] {$held=$true}
    if(-not $held){throw 'TideMotionGpuValidation is held; retry after the current validation finishes'}
    $psi=[Diagnostics.ProcessStartInfo]::new($Tool)
    foreach($arg in @('batch',$manifest,'--models',$Models)){$psi.ArgumentList.Add($arg)}
    $psi.UseShellExecute=$false;$psi.CreateNoWindow=$true
    $psi.RedirectStandardOutput=$true;$psi.RedirectStandardError=$true
    $process=[Diagnostics.Process]::new();$process.StartInfo=$psi
    $timer=[Diagnostics.Stopwatch]::StartNew();$process.Start()|Out-Null
    $stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync()
    [long]$peak=0
    while(-not $process.WaitForExit(250)){$process.Refresh();$peak=[Math]::Max($peak,$process.WorkingSet64)}
    $timer.Stop()
    $stdout.Result|Set-Content -LiteralPath (Join-Path $OutputDirectory 'stdout.log')
    $stderr.Result|Set-Content -LiteralPath (Join-Path $OutputDirectory 'stderr.log')
    if($process.ExitCode -ne 0){throw "Inference failed: $($stderr.Result)"}
    foreach($pair in @(@('cold.glb',183),@('warm.glb',183),@('history-prompts.glb',123),@('constrained.glb',183),@('long.glb',1800))){
        $file=Join-Path $OutputDirectory $pair[0]
        $info=(& $Tool inspect $file)|ConvertFrom-Json
        if($LASTEXITCODE -ne 0 -or $info.frames -ne $pair[1]){throw "Frame-count/GLB roundtrip failed: $file"}
    }
    @{seconds=$timer.Elapsed.TotalSeconds;peakWorkingSetBytes=$peak;threads=8;device='cpu'}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $OutputDirectory 'measurements.json')
    Write-Output "PASS native batch, history, changing prompts, path, long-form and GLB frame counts; seconds=$($timer.Elapsed.TotalSeconds) peakBytes=$peak"
}finally{if($held){$mx.ReleaseMutex()};$mx.Dispose()}
