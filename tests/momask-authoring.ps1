param(
    [Parameter(Mandatory=$true)][string]$Tool,
    [Parameter(Mandatory=$true)][string]$Models,
    [ValidateSet('cpu','gpu')][string]$Device='cpu',
    [string]$Work,
    [switch]$Stress
)
$ErrorActionPreference='Stop'
$toolPath=(Resolve-Path -LiteralPath $Tool).Path
$modelPath=(Resolve-Path -LiteralPath $Models).Path
if(!$Work){$Work=Join-Path (Split-Path -Parent $PSScriptRoot) ('.research/momask-authoring-'+[guid]::NewGuid().ToString('N'))}
if(Test-Path -LiteralPath $Work){throw 'Use a new isolated work directory'}
$workPath=(New-Item -ItemType Directory -Path $Work).FullName
$oldTemp=$env:TEMP;$oldTmp=$env:TMP
$env:TIDE_HIDDEN='1';$env:TEMP=(New-Item -ItemType Directory -Path (Join-Path $workPath 'temp')).FullName;$env:TMP=$env:TEMP
$mutex=[Threading.Mutex]::new($false,'TideMotionGpuValidation');$held=$false
try {
    if($Device -eq 'gpu'){while(!$held){try{$held=$mutex.WaitOne(1000)}catch [Threading.AbandonedMutexException]{$held=$true}}}
    $manifest=Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'momask-authoring.json') | ConvertFrom-Json -AsHashtable
    foreach($action in $manifest.actions){$action.device=$Device}
    if($Stress){$manifest.actions+=@{action='generate';backend='momask';prompt='A person walks forward.';duration=600;seed=7;output='maximum.glb'}}
    $manifestPath=Join-Path $workPath 'jobs.json'
    $manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath
    $start=[Diagnostics.ProcessStartInfo]::new($toolPath)
    $start.UseShellExecute=$false;$start.CreateNoWindow=$true;$start.RedirectStandardOutput=$true;$start.RedirectStandardError=$true
    foreach($argument in @('batch',$manifestPath,'--models',$modelPath)){$start.ArgumentList.Add($argument)}
    $process=[Diagnostics.Process]::new();$process.StartInfo=$start
    $watch=[Diagnostics.Stopwatch]::StartNew();$process.Start() | Out-Null
    $stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync();$peak=0L
    while(!$process.WaitForExit(100)){$process.Refresh();$peak=[Math]::Max($peak,$process.PeakWorkingSet64)}
    $watch.Stop();$stdout.Result | Set-Content -LiteralPath (Join-Path $workPath 'native.log');$stderr.Result | Set-Content -LiteralPath (Join-Path $workPath 'native-stderr.log')
    @{elapsedSeconds=$watch.Elapsed.TotalSeconds;sampledPeakWorkingSetBytes=$peak;device=$Device;stress=[bool]$Stress}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $workPath 'process.json')
    if($process.ExitCode -ne 0){throw 'Native MoMask authoring batch failed'}
    foreach($action in $manifest.actions){
        $file=Join-Path $workPath $action.output
        $info=& $toolPath inspect $file | ConvertFrom-Json
        if($LASTEXITCODE -ne 0 -or $info.frames -lt 5){throw "Invalid native asset: $file"}
        $recipe=Get-Content -Raw -LiteralPath ($file+'.json') | ConvertFrom-Json
        if($recipe.backend -ne 'momask' -or $recipe.device -ne $Device){throw 'Wrong backend/device'}
    }
    # A rejected candidate search must leave no asset or partial output.
    $noMatch=Join-Path $workPath 'no-match.glb'
    & $toolPath generate 'A person stands still in a relaxed pose.' --backend momask --device $Device --duration 3 --seed 0 --footsteps 64 --attempts 2 --models $modelPath -o $noMatch 2> (Join-Path $workPath 'no-match.log')
    if($LASTEXITCODE -eq 0 -or (Test-Path -LiteralPath $noMatch) -or (Test-Path -LiteralPath ($noMatch+'.part'))){throw 'Failed candidate selection wrote an asset'}
    if(!(Select-String -LiteralPath (Join-Path $workPath 'no-match.log') -Pattern 'No MoMask candidate matched')){throw 'Candidate failure had an unexpected cause'}
    Write-Output "Native MoMask authoring controls passed: $workPath"
} finally {
    $env:TEMP=$oldTemp;$env:TMP=$oldTmp
    if($held){$mutex.ReleaseMutex()};$mutex.Dispose()
}
exit 0
