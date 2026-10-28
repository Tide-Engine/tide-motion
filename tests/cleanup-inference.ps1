param(
    [Parameter(Mandatory=$true)][string]$Models,
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Target,
    [string]$Tool=(Join-Path $PSScriptRoot '../build/Release/tide-motion.exe'),
    [string]$Python='python'
)
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1'
$toolPath=(Resolve-Path -LiteralPath $Tool).Path
$modelPath=(Resolve-Path -LiteralPath $Models).Path
$work=Join-Path (Split-Path -Parent $PSScriptRoot) ('.research/cleanup-connect-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
$env:TEMP=Join-Path $work 'temp'
$env:TMP=$env:TEMP
New-Item -ItemType Directory -Path $env:TEMP | Out-Null
foreach($entry in @(@($Source,'source'),@($Target,'target'))) {
    Copy-Item -LiteralPath $entry[0] -Destination (Join-Path $work ($entry[1]+'.glb'))
    # Deliberately add interior authored pins, independently of generation metadata.
    $metadata=@{}
    if(Test-Path -LiteralPath ($entry[0]+'.json')) { $metadata=Get-Content -LiteralPath ($entry[0]+'.json') -Raw | ConvertFrom-Json -AsHashtable }
    $inherited=@()
    if($metadata.ContainsKey('protectedFrames')) {$inherited=@($metadata.protectedFrames)}
    $metadata.protectedFrames=$inherited+@(1)
    $metadata | ConvertTo-Json -Depth 32 | Set-Content (Join-Path $work ($entry[1]+'.glb.json'))
}
$info=& $toolPath inspect (Join-Path $work 'source.glb') | ConvertFrom-Json
if($LASTEXITCODE -or $info.frames -lt 4){throw 'Source must have at least four SOMA frames'}
@{contacts=@(@{foot='left-foot';start=0;end=([Math]::Min(15,$info.frames-1))},@{foot='right-foot';start=0;end=([Math]::Min(15,$info.frames-1))})} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $work 'contacts.json')
@{actions=@(
    @{action='connect';from='source.glb';to='target.glb';offset=@(0,0,0.4);tokens=8;threads=1;joined=$true;output='joined.glb'},
    @{action='connect';from='source.glb';to='target.glb';offset=@(0,0,0.4);tokens=8;threads=1;output='bridge.glb'},
    @{action='cleanup';source='joined.glb';contacts='contacts.json';output='cleaned.glb'}
)} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $work 'jobs.json')
# MotionBricks uses only ORT CPU here, one thread, and no benchmark timings.
# This does not acquire/use the GPU or contend for its validation mutex.
& $toolPath batch (Join-Path $work 'jobs.json') --models $modelPath *> (Join-Path $work 'connect.log')
if($LASTEXITCODE){throw "Connect/cleanup failed: $work/connect.log"}
if(@(Select-String -LiteralPath (Join-Path $work 'connect.log') -Pattern 'Loading MotionBricks SOMA').Count -ne 1){throw 'Expected one native ONNX session'}
foreach($name in @('joined','bridge')) {
    & $Python (Join-Path $PSScriptRoot 'verify_connect.py') (Join-Path $work 'source.glb') (Join-Path $work 'target.glb') (Join-Path $work ($name+'.glb'))
    if($LASTEXITCODE){throw 'Independent connect verification failed'}
}
& $Python (Join-Path $PSScriptRoot 'verify_cleanup.py') (Join-Path $work 'joined.glb') (Join-Path $work 'cleaned.glb') LeftFoot RightFoot
if($LASTEXITCODE){throw 'Independent cleanup verification failed'}
Write-Output "Native connected cleanup verified: $work"
