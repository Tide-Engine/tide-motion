param(
    [Parameter(Mandatory=$true)][string]$Tool,
    [Parameter(Mandatory=$true)][string]$Models,
    [string]$Tide,
    [ValidateRange(1,64)][int]$Seeds=8
)
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1'
$Tool=(Resolve-Path -LiteralPath $Tool).Path
$Models=(Resolve-Path -LiteralPath $Models).Path
$root=Split-Path -Parent $PSScriptRoot
$work=Join-Path $root ('.research/dart-validation-'+[guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($work)|Out-Null
$env:TEMP=$work;$env:TMP=$work
$actions=@()
foreach($seed in 1..$Seeds) {
    foreach($kind in @('goals','mixed','continuation')) {
        $segments=switch($kind) {
            'goals' { @(@{text='walk';chunks=64;target=@(0,2)},@{text='walk';chunks=64;target=@(2,2)},@{text='walk';chunks=64;target=@(2,0)}) }
            'mixed' { @(@{text='walk';chunks=64;target=@(0,2)},@{text='run';chunks=64;target=@(3,2)},@{text='hop on left leg';chunks=64;target=@(3,0)}) }
            'continuation' { @(@{text='walk';chunks=12},@{text='run';chunks=12},@{text='hop on left leg';chunks=12}) }
        }
        $name="$kind-$seed"
        @{version=1;seed=$seed;warmupChunks=8;segments=$segments}|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $work "$name.json")
        $actions+=@{action='dart';plan="$name.json";output="$name.glb";threads=4}
    }
}
@{actions=$actions}|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $work 'batch.json')
$info=[Diagnostics.ProcessStartInfo]::new($Tool)
$info.UseShellExecute=$false;$info.CreateNoWindow=$true;$info.RedirectStandardOutput=$true;$info.RedirectStandardError=$true
foreach($arg in @('batch',(Join-Path $work 'batch.json'),'--models',$Models)){$info.ArgumentList.Add($arg)}
$process=[Diagnostics.Process]::new();$process.StartInfo=$info
$watch=[Diagnostics.Stopwatch]::StartNew();[void]$process.Start()
$stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync();$peak=0L
while(!$process.WaitForExit(100)){$process.Refresh();$peak=[Math]::Max($peak,$process.WorkingSet64)}
$watch.Stop();$log=$stdout.Result+"`n"+$stderr.Result
[IO.File]::WriteAllText((Join-Path $work 'batch.log'),$log)
if($process.ExitCode){throw "DART batch failed: $log"}
if(([regex]::Matches($log,'Loading DART native')).Count -ne 1){throw 'DART sessions were not reused'}
$success=0;$goals=0
foreach($action in $actions){
    $recipe=Get-Content -Raw -LiteralPath ((Join-Path $work $action.output)+'.json')|ConvertFrom-Json
    foreach($segment in $recipe.metrics.segments){if($null -ne $segment.target){$goals++;if($segment.reached){$success++}}}
    & $Tool inspect (Join-Path $work $action.output)|Out-Null
    if($LASTEXITCODE){throw 'DART GLB failed native reimport'}
}
if($Tide){& $Tool preview (Join-Path $work 'goals-1.glb') --tide $Tide --capture (Join-Path $work 'goals.png') --frames 51;if($LASTEXITCODE){throw 'DART preview failed'}}
@{directory=$work;actions=$actions.Count;goals=$goals;reached=$success;seconds=$watch.Elapsed.TotalSeconds;peakWorkingSetBytes=$peak}|ConvertTo-Json|Tee-Object -FilePath (Join-Path $work 'summary.json')
