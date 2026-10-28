param([Parameter(Mandatory=$true)][string]$Archive)
$ErrorActionPreference='Stop'
$env:TIDE_HIDDEN='1'
$archivePath=(Resolve-Path -LiteralPath $Archive).Path
$repo=Split-Path -Parent $PSScriptRoot
$work=Join-Path $repo ('.research/upgrade-test-'+[guid]::NewGuid().ToString('N'))
$root=Join-Path $work "with spaces, 'quotes' and $([char]0x00E9)"
New-Item -ItemType Directory -Path (Join-Path $root 'models') -Force | Out-Null
$archiveFolder=Join-Path $work "archive $([char]0x00E9)"
New-Item -ItemType Directory -Path $archiveFolder | Out-Null
$copiedArchive=Join-Path $archiveFolder ([IO.Path]::GetFileName($archivePath))
Copy-Item -LiteralPath $archivePath -Destination $copiedArchive
Copy-Item -LiteralPath ($archivePath+'.sha256') -Destination ($copiedArchive+'.sha256')
$archivePath=$copiedArchive
$stage=Join-Path $work 'bootstrap'
Expand-Archive -LiteralPath $archivePath -DestinationPath $stage
$tool=Get-ChildItem -LiteralPath $stage -Filter tide-motion.exe -Recurse | Select-Object -First 1 -ExpandProperty FullName
if(!$tool){throw 'Package has no executable'}
Set-Content -LiteralPath (Join-Path $root 'models/keep.txt') -Value 'keep model'
Set-Content -LiteralPath (Join-Path $root 'settings.json') -Value 'keep settings'
function Invoke-Upgrade([string[]]$Arguments) {
    & $tool @Arguments *> (Join-Path $work 'output.log')
    return $LASTEXITCODE
}
function Refuses([string]$Match,[string[]]$Arguments) {
    $code=Invoke-Upgrade $Arguments
    $text=Get-Content -LiteralPath (Join-Path $work 'output.log') -Raw
    if(!$code -or $text -notmatch $Match){throw "Expected refusal $Match, got $code : $text"}
}
if(Invoke-Upgrade @('upgrade','--archive',$archivePath,'--check','--root',(Join-Path $work 'untouched'))){throw 'Archive check failed'}
if(Test-Path -LiteralPath (Join-Path $work 'untouched')){throw '--check modified destination'}
if(Invoke-Upgrade @('upgrade','--archive',$archivePath,'--root',$root,'--force')){Get-Content (Join-Path $work 'output.log');throw 'Offline upgrade failed'}
$tool=Join-Path $root 'bin/tide-motion.exe'
Set-Content -LiteralPath (Join-Path $root 'README.md') -Value 'no-op sentinel'
if(Invoke-Upgrade @('upgrade','--archive',$archivePath)){throw 'Equal-version check failed'}
if((Get-Content (Join-Path $root 'README.md')) -ne 'no-op sentinel'){throw 'Equal-version upgrade replaced files without --force'}
# Only explicitly committed old backups may be collected.
foreach($state in @('pending','committed')) {
    $old=Join-Path $root ('.upgrade-backup-'+$state)
    New-Item -ItemType Directory -Path $old | Out-Null
    [IO.File]::WriteAllText((Join-Path $old 'marker'),('tide-motion-upgrade-v1'+$(if($state -eq 'committed'){'-committed'}else{''})))
}
# Upgrade from the running installed executable, with its EXE/DLL mappings live.
if(Invoke-Upgrade @('upgrade','--archive',$archivePath,'--force')){Get-Content (Join-Path $work 'output.log');throw 'Live self-upgrade failed'}
if(Test-Path -LiteralPath (Join-Path $root '.upgrade-backup-committed')){throw 'Committed backup was not collected'}
if(!(Test-Path -LiteralPath (Join-Path $root '.upgrade-backup-pending'))){throw 'Uncommitted recovery backup was deleted'}
foreach($item in @('install.ps1','install.sh','licenses/META-LLAMA-3.txt','README.md','PROMPTING.md')){
    if(!(Test-Path -LiteralPath (Join-Path $root $item))){throw "Upgrade omitted $item"}
}
if((Get-Content (Join-Path $root 'models/keep.txt')) -ne 'keep model' -or (Get-Content (Join-Path $root 'settings.json')) -ne 'keep settings'){throw 'Upgrade changed models/settings'}
$before=(Get-FileHash -LiteralPath $tool).Hash
# Failure after earlier entries have been swapped must restore all old entries.
Set-Content -LiteralPath (Join-Path $root 'notes/custom-sentinel.txt') -Value 'rollback note'
Set-Content -LiteralPath (Join-Path $root 'README.md') -Value 'rollback README'
$handle=[IO.File]::Open((Join-Path $root 'LICENSE'),[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
try { Refuses 'filesystem error|rename|denied' @('upgrade','--archive',$archivePath,'--force') } finally {$handle.Dispose()}
if((Get-Content (Join-Path $root 'README.md')) -ne 'rollback README' -or (Get-Content (Join-Path $root 'notes/custom-sentinel.txt')) -ne 'rollback note'){throw 'Failed upgrade did not roll back metadata'}
$handle=[IO.File]::Open((Join-Path $root '.upgrade.lock'),[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::None)
try { Refuses 'Another upgrade' @('upgrade','--archive',$archivePath,'--force') } finally {$handle.Dispose()}
$badDir=Join-Path $work 'bad'
New-Item -ItemType Directory -Path $badDir | Out-Null
$bad=Join-Path $badDir ([IO.Path]::GetFileName($archivePath))
Copy-Item -LiteralPath $archivePath -Destination $bad
Set-Content -LiteralPath "$bad.sha256" -Value ('0'*64)
Refuses 'checksum mismatch' @('upgrade','--archive',$bad,'--force')
# A valid hash does not make traversal safe.
Remove-Item -LiteralPath $bad
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip=[IO.Compression.ZipFile]::Open($bad,[IO.Compression.ZipArchiveMode]::Create)
try {$zip.CreateEntry('../escape.txt') | Out-Null} finally {$zip.Dispose()}
Set-Content -LiteralPath "$bad.sha256" -Value (Get-FileHash -LiteralPath $bad).Hash
Refuses 'Unsafe archive entry' @('upgrade','--archive',$bad,'--force')
if(Test-Path -LiteralPath (Join-Path $root 'escape.txt')){throw 'Traversal wrote outside stage'}
# A runnable package must also identify itself as the requested release.
$wrong=Join-Path $badDir 'tide-motion-999.0.0-windows-x64.zip'
Copy-Item -LiteralPath $archivePath -Destination $wrong
Set-Content -LiteralPath "$wrong.sha256" -Value (Get-FileHash -LiteralPath $wrong).Hash
Refuses 'version does not match' @('upgrade','--archive',$wrong,'--force')
Refuses 'Version must be' @('upgrade','--version','1.2.3/evil')
Refuses 'Unknown upgrade option' @('upgrade','--models','no-models')
if((Get-FileHash -LiteralPath $tool).Hash -ne $before){throw 'Failed upgrade changed executable'}
Write-Output "Offline/live upgrade, preservation, rollback, locking and refusal checks passed: $work"
exit 0
