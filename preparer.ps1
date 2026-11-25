# Prépare une copie de travail pour un essai : compile (normale et diagnostic), traduit les shaders dans le cache,
# puis prépare les pipelines en chassant les plantages du compilateur AMD (liste refus-amd.txt).
param([string]$Src = 'D:\victory-src\main2', [int]$MaxPlantages = 100, [string]$Liste = 'D:\victory-data\scripts\refus-amd.txt')
$g = 'D:\GTA6.Xbox.v1.0.2.1'
$fichiers = "$g\shaders.rpf", "$g\game_scarlett_master.exe", "$g\XboxBvhBuilder.dll", "$g\amd_fidelityfx_upscaler_xbox.dll"
Set-Location $Src
$t0 = Get-Date
cargo build --release --quiet -p victory 2>&1 | Select-String '^error' -Context 0, 5 | Select-Object -First 3
if ($LASTEXITCODE -ne 0) { "ÉCHEC de compilation (normale)"; exit 1 }
cargo build --profile diag --quiet -p victory --features diagnostics 2>&1 | Select-String '^error' -Context 0, 5 | Select-Object -First 3
if ($LASTEXITCODE -ne 0) { "ÉCHEC de compilation (diagnostic)"; exit 1 }
"Compilé en $([int]((Get-Date) - $t0).TotalMinutes) min"
$t0 = Get-Date
.\target\diag\victory.exe translate --cache @fichiers *> D:\victory-data\preparer-traduction.txt
"Traduction : code $LASTEXITCODE en $([int]((Get-Date) - $t0).TotalSeconds) s ; $((Get-Content D:\victory-data\preparer-traduction.txt -Tail 1))"
# Chasse : tous les fils, et à chaque plantage un passage sur un seul fil pour nommer le pipeline fautif.
$liste = $Liste
if (-not (Test-Path $liste)) { New-Item -ItemType File $liste | Out-Null }
$log = 'D:\victory-data\pipelines-log.txt'
function Invoke-Prebuild([string]$Jobs) {
    $env:VICTORY_REFUSE_SHADERS = (Get-Content $liste | Where-Object { $_ -match '^[0-9a-f]{16}$' }) -join ','
    $env:VICTORY_PIPELINE_LOG = $log
    Remove-Item $log -ErrorAction SilentlyContinue
    $a = @('translate', '--cache', '--pipelines') + $(if ($Jobs) { @('--jobs', $Jobs) } else { @() }) + $fichiers
    & .\target\diag\victory.exe @a *> D:\victory-data\preparer-pipelines.txt
    $LASTEXITCODE
}
$t0 = Get-Date
$nouveaux = 0
for ($i = 1; $i -le $MaxPlantages; $i++) {
    if ((Invoke-Prebuild '') -eq 0) { break }
    if ((Invoke-Prebuild '1') -eq 0) { break }
    $start = Get-Content $log | Where-Object { $_ -like 'start*' } | Select-Object -Last 1
    $built = Get-Content $log | Where-Object { $_ -like 'built*' } | ForEach-Object { ($_ -split "`t")[1] }
    if (-not $start -or ($built -contains ($start -split "`t")[1])) { "Plantage sans pipeline identifiable"; break }
    [regex]::Matches(($start -split "`t")[2], '[0-9a-f]{16}') | ForEach-Object { $_.Value } | Add-Content $liste
    $nouveaux++
}
$env:VICTORY_REFUSE_SHADERS = ''
$env:VICTORY_PIPELINE_LOG = ''
"Pipelines préparés en $([int]((Get-Date) - $t0).TotalMinutes) min ; $nouveaux nouveaux shaders refusés ; liste : $(@(Get-Content $liste | Where-Object { $_ -match '^[0-9a-f]{16}$' }).Count)"
Get-Content D:\victory-data\preparer-pipelines.txt | Select-String 'skipped|refused|pipelines to make' | Select-Object -Last 3 | ForEach-Object { "  $($_.Line.Trim())" }
