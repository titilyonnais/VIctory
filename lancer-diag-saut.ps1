# Lance la version de diagnostic main + #1544 + #575 + #635 avec la couche de débogage D3D12, DRED et la mesure du temps.
# Cache officiel (shaders traduits par main). Journal : D:\victory-data\diag-saut.txt.
$journal = 'D:\victory-data\diag-saut.txt'
$env:VICTORY_CACHE_DIR = ''
$env:VICTORY_D3D12_DEBUG = '1'
$env:VICTORY_D3D12_DRED = '1'
$env:VICTORY_GPU_TIMING = '1'
$env:VICTORY_SKIP_PIPELINES = '9146291b38ea9ce1'
try {
    Set-Location 'C:\Users\<utilisateur>\Desktop\claude-projects\VIctory-essai'
    & .\target\diag\victory.exe run 'D:\GTA6.Xbox.v1.0.2.1\game_scarlett_master.exe' --execute 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath $journal
} finally {
    $env:VICTORY_D3D12_DEBUG = ''
    $env:VICTORY_D3D12_DRED = ''
    $env:VICTORY_GPU_TIMING = ''
    $env:VICTORY_SKIP_PIPELINES = ''
    Set-Location 'C:\Users\<utilisateur>\Desktop\claude-projects\VIctory'
}
