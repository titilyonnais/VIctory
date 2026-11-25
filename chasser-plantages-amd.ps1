# Prépare les pipelines (tous les fils), et à chaque plantage du compilateur AMD reprend sur un seul fil :
# la bibliothèque garde ce qui est fait, donc le premier pipeline qu'il compile est celui qui plante.
# Ses shaders vont dans la liste de refus (D:\victory-data\scripts\refus-amd.txt). Au plus $Max plantages.
param([int]$Max = 25)
$g = 'D:\GTA6.Xbox.v1.0.2.1'
$liste = 'D:\victory-data\scripts\refus-amd.txt'
$log = 'D:\victory-data\pipelines-log.txt'
if (-not (Test-Path $liste)) { Set-Content $liste '' }
Set-Location 'D:\victory-src\nouveau'
$fichiers = "$g\shaders.rpf", "$g\game_scarlett_master.exe", "$g\XboxBvhBuilder.dll", "$g\amd_fidelityfx_upscaler_xbox.dll"
function Invoke-Prebuild([string]$Jobs) {
    $env:VICTORY_REFUSE_SHADERS = (Get-Content $liste | Where-Object { $_ -match '^[0-9a-f]{16}$' }) -join ','
    $env:VICTORY_PIPELINE_LOG = $log
    Remove-Item $log -ErrorAction SilentlyContinue
    $a = @('translate', '--cache', '--pipelines') + $(if ($Jobs) { @('--jobs', $Jobs) } else { @() }) + $fichiers
    & .\target\diag\victory.exe @a *> D:\victory-data\translate-chasse.txt
    $LASTEXITCODE
}
for ($i = 1; $i -le $Max; $i++) {
    $t0 = Get-Date
    $code = Invoke-Prebuild ''
    if ($code -eq 0) { "Terminé sans plantage ($([int]((Get-Date) - $t0).TotalSeconds) s)"; break }
    $code = Invoke-Prebuild '1'
    if ($code -eq 0) { "Terminé sur un fil, sans plantage"; break }
    $start = Get-Content $log | Where-Object { $_ -like 'start*' } | Select-Object -Last 1
    $built = Get-Content $log | Where-Object { $_ -like 'built*' } | ForEach-Object { ($_ -split "`t")[1] }
    if (-not $start -or ($built -contains ($start -split "`t")[1])) { "Plantage $i (code $code) sans pipeline en cours identifiable"; break }
    $shaders = ($start -split "`t")[2]
    "Plantage $i : $shaders"
    [regex]::Matches($shaders, '[0-9a-f]{16}') | ForEach-Object { $_.Value } | Add-Content $liste
}
$env:VICTORY_REFUSE_SHADERS = ''
$env:VICTORY_PIPELINE_LOG = ''
"Liste de refus : $((Get-Content $liste | Where-Object { $_ -match '^[0-9a-f]{16}$' }) -join ',')"
Get-Content D:\victory-data\translate-chasse.txt -Tail 2
