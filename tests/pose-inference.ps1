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
@{actions=@(@{action='pose-sequence';recipe='poses.json';threads=8;output='authored.glb'},@{action='connect';from='idle.glb';to='walk.glb';tokens=8;threads=8;output='compatible.glb'})} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $work 'jobs.json')
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
    if($Tide) {
        & $Tool preview (Join-Path $work 'authored.glb') --tide $Tide --capture (Join-Path $work 'arrival.png') --frames 51
        if($LASTEXITCODE){throw 'Bounded Tide preview failed'}
    }
} finally {if($locked){$mutex.ReleaseMutex()};$mutex.Dispose()}
Write-Output "Pose authoring inference validated: $work"
