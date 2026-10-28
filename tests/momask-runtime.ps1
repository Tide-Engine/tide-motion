param(
    [Parameter(Mandatory=$true)][string]$Tool,
    [Parameter(Mandatory=$true)][string]$Models,
    [string]$Work
)
$ErrorActionPreference='Stop'
$toolPath=(Resolve-Path -LiteralPath $Tool).Path
$modelPath=(Resolve-Path -LiteralPath $Models).Path
if(!$Work){$Work=Join-Path (Split-Path -Parent $PSScriptRoot) ('.research/momask-runtime-'+[guid]::NewGuid().ToString('N'))}
if(Test-Path -LiteralPath $Work){throw 'Use a new isolated work directory'}
New-Item -ItemType Directory -Path $Work | Out-Null
$workPath=(Resolve-Path -LiteralPath $Work).Path
$env:TIDE_HIDDEN='1'
$savedPath=$env:PATH
$mutex=[Threading.Mutex]::new($false,'TideMotionGpuValidation');$held=$false
try {
    while(!$held){try{$held=$mutex.WaitOne(1000)}catch [Threading.AbandonedMutexException]{$held=$true}}
    $actions=@(
        @{action='generate';backend='momask';device='gpu';rig='soma77';prompt='A person stands still in a relaxed pose.';duration=3;seed=2;output='idle77.glb'},
        @{action='mirror';source='idle77.glb';output='mirror77.glb'},
        @{action='inpaint';device='gpu';prompt='A person waves with the right hand.';source='idle77.glb';begin=18;end=66;seed=3;output='wave77.glb'}
    )
    $manifest=Join-Path $workPath 'jobs.json'
    @{actions=$actions}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $manifest
    $env:PATH="$env:SystemRoot/System32;$env:SystemRoot"
    $stderr=Join-Path $workPath 'batch.log'
    & $toolPath batch $manifest --models $modelPath 2> $stderr
    if($LASTEXITCODE -ne 0){throw 'Packaged native GPU/inpaint batch failed'}
    if(@(Get-Content -LiteralPath $stderr|Select-String 'Loaded MoMask bundle once').Count -ne 1){throw 'MoMask sessions were not reused'}
    foreach($name in @('idle77','mirror77','wave77')){
        $file=Join-Path $workPath ($name+'.glb')
        $info=& $toolPath inspect $file | ConvertFrom-Json
        if($LASTEXITCODE -ne 0 -or $info.rig -ne 'soma77' -or $info.frames -ne 89){throw 'Packaged SOMA-77 GLB check failed'}
        if($name -ne 'mirror77'){
            $recipe=Get-Content -Raw -LiteralPath ($file+'.json')|ConvertFrom-Json
            if($recipe.device -ne 'gpu' -or $recipe.accelerator -ne 'directml'){throw 'GPU was not used'}
        }
    }
    # Simulate a deployment lacking the optional GPU DLL directory.
    $cpu=Join-Path $workPath 'without GPU';New-Item -ItemType Directory -Path $cpu | Out-Null
    Copy-Item -LiteralPath $toolPath -Destination (Join-Path $cpu 'tide-motion.exe')
    Get-ChildItem -LiteralPath (Split-Path -Parent $toolPath) -Filter '*.dll' | Copy-Item -Destination $cpu
    $cpuTool=Join-Path $cpu 'tide-motion.exe'
    $strict=Join-Path $workPath 'strict.glb'
    $strictLog=Join-Path $workPath 'strict.log'
    & $cpuTool generate 'A person walks forward.' --backend momask --device gpu --duration 1 --models $modelPath -o $strict 2> $strictLog
    if($LASTEXITCODE -eq 0 -or (Test-Path -LiteralPath $strict) -or !(Select-String -LiteralPath $strictLog -Pattern 'MoMask GPU:')){throw 'Missing GPU strict refusal failed'}
    $auto=Join-Path $workPath 'auto.glb'
    & $cpuTool generate 'A person walks forward.' --backend momask --device auto --duration 1 --models $modelPath -o $auto 2> (Join-Path $workPath 'auto.log')
    if($LASTEXITCODE -ne 0){throw 'Actual auto CPU inference failed'}
    $recipe=Get-Content -Raw -LiteralPath ($auto+'.json')|ConvertFrom-Json
    if($recipe.device -ne 'cpu'){throw 'Auto fallback reports wrong device'}
    foreach($relative in @('ardy-ort-directml/LICENSE','ardy-ort-directml/ThirdPartyNotices.txt','directml/LICENSE.txt','directml/ThirdPartyNotices.txt')){
        if(!(Test-Path -LiteralPath (Join-Path (Split-Path -Parent (Split-Path -Parent $toolPath)) ('licenses/'+$relative)))){throw "Missing runtime notice: $relative"}
    }
    Write-Output "MoMask native packaged GPU, session reuse, SOMA-77 edit, strict refusal and CPU fallback pass: $workPath"
} finally {
    $env:PATH=$savedPath
    if($held){$mutex.ReleaseMutex()};$mutex.Dispose()
}
