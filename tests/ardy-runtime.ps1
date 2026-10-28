param([Parameter(Mandatory=$true)][string]$Tool,
      [Parameter(Mandatory=$true)][string]$Models,
      [Parameter(Mandatory=$true)][string]$OutputDirectory)
# Opt-in native runtime/package smoke. External, pinned ARDY + text models are
# required; no weights or generated assets belong in the source distribution.
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1'
$Tool=(Resolve-Path -LiteralPath $Tool).Path
$Models=(Resolve-Path -LiteralPath $Models).Path
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory|Out-Null
$package=Split-Path -Parent (Split-Path -Parent $Tool)
foreach($notice in @('ardy-ort-directml/LICENSE','ardy-ort-directml/ThirdPartyNotices.txt','directml/LICENSE.txt','directml/ThirdPartyNotices.txt')){
    if(-not (Test-Path -LiteralPath (Join-Path $package "licenses/$notice"))){throw "Packaged GPU runtime omitted $notice"}
}
function Check-Asset([string]$File,[string]$Device,[int]$Frames){
    $info=(& $Tool inspect $File)|ConvertFrom-Json
    if($LASTEXITCODE -ne 0 -or $info.frames -ne $Frames){throw "Invalid native GLB: $File"}
    $recipe=Get-Content -LiteralPath "$File.json" -Raw|ConvertFrom-Json
    if($recipe.device -ne $Device){throw "Wrong actual device: $File"}
}
$mx=[Threading.Mutex]::new($false,'TideMotionGpuValidation');$held=$false
$savedPath=$env:PATH
try{
    try{$held=$mx.WaitOne(0)}catch [Threading.AbandonedMutexException]{$held=$true}
    if(-not $held){throw 'TideMotionGpuValidation is held; retry after the current validation finishes'}
    # The installed executable must not depend on Python, CUDA, a build SDK or PATH.
    $env:PATH="$env:SystemRoot/System32;$env:SystemRoot"
    $gpu=Join-Path $OutputDirectory 'gpu.glb'
    & $Tool generate 'A person walks forward.' --backend ardy --duration .2 --steps 1 --threads 8 --no-correction --models $Models -o $gpu 2> (Join-Path $OutputDirectory 'gpu.log')
    if($LASTEXITCODE -ne 0){throw 'Packaged default-device GPU inference failed'}
    Check-Asset $gpu gpu 6
    if((Get-Content -LiteralPath "$gpu.json" -Raw|ConvertFrom-Json).accelerator -ne 'directml'){throw 'Missing GPU provenance'}
    # Keep the two ORT versions alive in one process: GPU ARDY followed by CPU
    # MotionBricks. This guards accidental process-global API-table interposition.
    if(Test-Path -LiteralPath (Join-Path $Models 'motionbricks.onnx')){
        $jobs=Join-Path $OutputDirectory 'mixed.json'
        @{actions=@(
            @{action='generate';backend='ardy';prompt='A person walks forward.';duration=1;steps=1;threads=8;correct=$false;output='mixed-source.glb'},
            @{action='connect';from='mixed-source.glb';to='mixed-source.glb';threads=8;output='mixed-bridge.glb'}
        )}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $jobs -Encoding utf8
        & $Tool batch $jobs --models $Models 2> (Join-Path $OutputDirectory 'mixed.log')
        if($LASTEXITCODE -ne 0){throw 'Mixed GPU/CPU runtime inference failed'}
        Check-Asset (Join-Path $OutputDirectory 'mixed-source.glb') gpu 30
    }
    # A private incomplete deployment tests fallback without altering the build
    # or installed runtime. Explicit gpu must fail; auto must really generate CPU.
    $incomplete=Join-Path $OutputDirectory 'without DirectML'
    New-Item -ItemType Directory -Path $incomplete|Out-Null
    Copy-Item -LiteralPath $Tool -Destination $incomplete
    Get-ChildItem -LiteralPath (Split-Path -Parent $Tool) -Filter '*.dll'|Copy-Item -Destination $incomplete
    $copy=Join-Path $incomplete (Split-Path -Leaf $Tool)
    $strict=Join-Path $OutputDirectory 'strict.glb'
    & $copy generate 'A person walks forward.' --backend ardy --device gpu --duration .2 --steps 1 --threads 8 --no-correction --models $Models -o $strict 2> (Join-Path $OutputDirectory 'strict.log')
    if($LASTEXITCODE -eq 0 -or (Test-Path -LiteralPath $strict)){throw 'Explicit GPU silently fell back'}
    if((Get-Content -LiteralPath (Join-Path $OutputDirectory 'strict.log') -Raw) -notmatch 'ARDY GPU: cannot load'){throw 'Strict GPU failed for an unexpected reason'}
    $fallback=Join-Path $OutputDirectory 'fallback.glb'
    & $copy generate 'A person walks forward.' --backend ardy --duration .2 --steps 1 --threads 8 --no-correction --models $Models -o $fallback 2> (Join-Path $OutputDirectory 'fallback.log')
    if($LASTEXITCODE -ne 0){throw 'Auto CPU fallback failed'}
    Check-Asset $fallback cpu 6
    Write-Output 'PASS default-device GPU in CLI/batch, isolated runtime coexistence (when MotionBricks supplied), strict-device failure and default-device CPU fallback'
}finally{$env:PATH=$savedPath;if($held){$mx.ReleaseMutex()};$mx.Dispose()}
