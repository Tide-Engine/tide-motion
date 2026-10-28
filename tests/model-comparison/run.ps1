param([string]$Tool,[string]$Models,[string]$Work)
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1';$env:KIMODO_PROFILE='1'
$Tool=(Resolve-Path $Tool).Path;$Models=(Resolve-Path $Models).Path;$Work=(Resolve-Path $Work).Path
@{executableSha256=(Get-FileHash $Tool -Algorithm SHA256).Hash;tool=$Tool;models=$Models}|ConvertTo-Json|Set-Content "$Work/executable.json"
$m=[Threading.Mutex]::new($false,'TideMotionGpuValidation');$held=$false;$p=$null
try {
 try {$held=$m.WaitOne(45000)} catch [Threading.AbandonedMutexException] {$held=$true}
 if(!$held){throw 'GPU validation mutex busy; manifests are ready for another run'}
 foreach($block in 1..4) {
  $p=$null;$start=Get-Date;$peak=[long]0;$other=$false
  $p=Start-Process $Tool -ArgumentList @('batch',"`"$Work/block$block.json`"",'--models',"`"$Models`"") -PassThru -WindowStyle Hidden -RedirectStandardOutput "$Work/block$block.out.log" -RedirectStandardError "$Work/block$block.err.log"
  while(!$p.WaitForExit(100)) {
   $p.Refresh();$peak=[Math]::Max([long]$peak,[long]$p.WorkingSet64)
   if(@(Get-Process tide-motion -ErrorAction SilentlyContinue|Where-Object {$_.Id -ne $p.Id}).Count){$other=$true}
  }
  $record=@{block=$block;seconds=((Get-Date)-$start).TotalSeconds;peakWorkingSetBytes=$peak;otherTideMotionObserved=$other;exitCode=$p.ExitCode}
  $record|ConvertTo-Json|Set-Content "$Work/block$block.resources.json"
  if($p.ExitCode -ne 0){throw "Block $block failed; see log"}
  Write-Output ($record|ConvertTo-Json -Compress)
 }
} finally {
 if($p -and !$p.HasExited){Stop-Process -Id $p.Id;$p.WaitForExit()}
 if($held){$m.ReleaseMutex()};$m.Dispose()
}
