# Lancement sans personne au clavier, pour la nuit :
# - Entrée toutes les 3 s de 15 à 180 s (écran « A PLAYER », écran titre) ;
# - images du monde enregistrées par le jeu (0, 30, 90, 180, 300 images après la première image de 500 dessins) ;
# - captures d'écran et arrêt au bout de 15 min : surveiller.ps1, lancé à part ;
# Version de diagnostic du dossier $Src, avec DRED, la liste de refus AMD, une limite de boucles des shaders
# et des pipelines sautés (ceux qu'un testeur RX 9070 XT devait sauter, issue #1528).
param(
    [string]$Src = 'D:\victory-src\main2',
    [string]$Nom = 'auto',
    [int]$Minutes = 15,
    [string]$LoopLimit = '256',
    [string]$Skip = 'eeed82b877b44b5f,fc5e66f8f873d3bb',
    [string]$PipelineThreads = '',
    [string]$Liste = 'D:\victory-data\scripts\refus-amd.txt'
)
$journal = "D:\victory-data\$Nom.txt"
$images = "D:\victory-data\images\$Nom"
New-Item -ItemType Directory -Force $images | Out-Null
Get-ChildItem $images -File -ErrorAction SilentlyContinue | Remove-Item
$env:VICTORY_CACHE_DIR = ''
$env:VICTORY_D3D12_DRED = '1'
$env:VICTORY_REFUSE_SHADERS = (Get-Content $Liste -ErrorAction SilentlyContinue | Where-Object { $_ -match '^[0-9a-f]{16}$' }) -join ','
$env:VICTORY_KEYS = ((15..180 | Where-Object { $_ % 3 -eq 0 }) | ForEach-Object { "${_}:Enter" }) -join ','
$env:VICTORY_CAPTURE = $images
$env:VICTORY_CAPTURE_FRAMES = '1'
$env:VICTORY_CAPTURE_WORLD = '0,90,300,450,600,800,1000'
$env:VICTORY_SHADER_LOOP_LIMIT = $LoopLimit
$env:VICTORY_SKIP_PIPELINES = $Skip
$env:VICTORY_PIPELINE_THREADS = $PipelineThreads
# Une ligne par pipeline créé, utilisé, fabriqué ou chargé de la bibliothèque.
$env:VICTORY_PIPELINE_LOG = "D:\victory-data\$Nom-pipelines.txt"
try {
    Set-Location $Src
    & .\target\diag\victory.exe run 'D:\GTA6.Xbox.v1.0.2.1\game_scarlett_master.exe' --execute 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath $journal
} finally {
    foreach ($v in 'VICTORY_D3D12_DRED', 'VICTORY_REFUSE_SHADERS', 'VICTORY_KEYS', 'VICTORY_CAPTURE', 'VICTORY_CAPTURE_FRAMES',
        'VICTORY_CAPTURE_WORLD', 'VICTORY_SHADER_LOOP_LIMIT', 'VICTORY_SKIP_PIPELINES', 'VICTORY_PIPELINE_LOG', 'VICTORY_PIPELINE_THREADS') {
        Set-Item "env:$v" ''
    }
    Set-Location 'C:\Users\<utilisateur>\Desktop\claude-projects\VIctory'
}
