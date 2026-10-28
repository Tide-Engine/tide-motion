param([Parameter(Mandatory=$true)][string]$Tool,[Parameter(Mandatory=$true)][string]$Models)
$ErrorActionPreference='Stop';$env:TIDE_HIDDEN='1'
$Tool=(Resolve-Path -LiteralPath $Tool).Path;$Models=(Resolve-Path -LiteralPath $Models).Path
$work=Join-Path (Split-Path -Parent $PSScriptRoot) ('.research/dart-runtime-'+[guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory($work)|Out-Null
$plan=Join-Path $work 'plan.json';@{version=1;segments=@(@{text='walk';chunks=1})}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath $plan
$mx=[Threading.Mutex]::new($false,'TideMotionGpuValidation');$held=$false;$savedPath=$env:PATH
try {
 try{$held=$mx.WaitOne(0)}catch [Threading.AbandonedMutexException] {$held=$true}
 if(!$held){throw 'GPU mutex busy; retry after validation completes'}
 $env:PATH="$env:SystemRoot/System32;$env:SystemRoot"
 & $Tool dart $plan --models $Models -o "$work/auto.glb" 2> "$work/auto.log"
 if($LASTEXITCODE){throw 'Packaged automatic GPU generation failed'}
 if((Get-Content "$work/auto.glb.json" -Raw|ConvertFrom-Json).device -ne 'gpu'){throw 'Default did not select GPU'}
 $incomplete=Join-Path $work 'without GPU runtime';[IO.Directory]::CreateDirectory($incomplete)|Out-Null
 Copy-Item -LiteralPath $Tool -Destination "$incomplete/tide-motion.exe"
 Copy-Item -LiteralPath (Join-Path (Split-Path -Parent $Tool) 'onnxruntime.dll') -Destination $incomplete
 & "$incomplete/tide-motion.exe" dart $plan --device gpu --models $Models -o "$work/strict.glb" 2> "$work/strict.log"
 if($LASTEXITCODE -eq 0 -or (Test-Path "$work/strict.glb")){throw 'Strict GPU silently fell back'}
 if((Get-Content "$work/strict.log" -Raw) -notmatch 'DART GPU: cannot load'){throw 'Unexpected strict GPU failure'}
 & "$incomplete/tide-motion.exe" dart $plan --models $Models -o "$work/fallback.glb" 2> "$work/fallback.log"
 if($LASTEXITCODE){throw 'Automatic CPU fallback failed'}
 if((Get-Content "$work/fallback.glb.json" -Raw|ConvertFrom-Json).device -ne 'cpu'){throw 'Fallback provenance incorrect'}
 foreach($name in @('auto','fallback')){& $Tool inspect "$work/$name.glb"|Out-Null;if($LASTEXITCODE){throw 'Runtime asset invalid'}}
 Write-Output "DART packaged GPU/default, strict refusal and actual CPU fallback passed: $work"
}finally{$env:PATH=$savedPath;if($held){$mx.ReleaseMutex()};$mx.Dispose()}
