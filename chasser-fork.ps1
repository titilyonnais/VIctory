# Chasse en jeu des shaders qui font planter le compilateur AMD dans le fork : lance le jeu (un seul fil de
# compilation, journal de début de compilation), et à chaque plantage amdxc64 ajoute les shaders du dernier
# pipeline commencé à refus-amd.txt, puis relance. S'arrête quand un essai dure $Minutes minutes sans ce plantage.
param([int]$Max = 8, [int]$Minutes = 12)
$liste = 'D:\victory-data\scripts\refus-amd.txt'
$log = 'D:\victory-data\fork-start.txt'
for ($i = 1; $i -le $Max; $i++) {
    $nom = "fork-chasse$i"
    Remove-Item $log -ErrorAction SilentlyContinue
    $env:VICTORY_PIPELINE_START_LOG = $log
    $jeu = Start-Process pwsh -PassThru -WindowStyle Minimized -ArgumentList @(
        '-NoProfile', '-File', 'D:\victory-data\scripts\lancer-auto.ps1',
        '-Src', 'D:\victory-src\fork', '-Nom', $nom, '-PipelineThreads', '1')
    $debut = Get-Date
    Start-Sleep -Seconds 20
    while ((& 'D:\victory-data\scripts\jeu.ps1') -and ((Get-Date) - $debut).TotalMinutes -lt $Minutes) { Start-Sleep -Seconds 10 }
    $duree = [int]((Get-Date) - $debut).TotalSeconds
    if (& 'D:\victory-data\scripts\jeu.ps1') {
        & 'D:\victory-data\scripts\capture-ecran.ps1' -Nom "$nom-fin" | Out-Null
        & 'D:\victory-data\scripts\jeu.ps1' -Arreter | Out-Null
        "Essai $i : a tenu $Minutes min sans plantage du compilateur"
        break
    }
    Start-Sleep -Seconds 3
    $fin = Get-Content "D:\victory-data\$nom.txt" -Tail 400 -ErrorAction SilentlyContinue | Select-String 'amdxc64.dll\+0x861b9f' | Select-Object -First 1
    if (-not $fin) { "Essai $i : arrêt après $duree s, sans plantage du compilateur AMD (voir $nom.txt)"; break }
    $start = Get-Content $log -ErrorAction SilentlyContinue | Select-Object -Last 1
    $hashes = [regex]::Matches("$start", '[0-9a-f]{16}') | ForEach-Object { $_.Value }
    if (-not $hashes) { "Essai $i : plantage AMD après $duree s, pipeline en cours inconnu"; break }
    "Essai $i : plantage AMD après $duree s sur $($start -replace '^start\t', '')"
    $hashes | Add-Content $liste
}
$env:VICTORY_PIPELINE_START_LOG = ''
"Liste de refus : $(@(Get-Content $liste | Where-Object { $_ -match '^[0-9a-f]{16}$' }).Count) shaders"
