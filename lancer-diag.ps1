# Lance la version de diagnostic (main + #1632 + #1418) avec la couche de débogage D3D12 et DRED.
# Le journal complet est copié dans D:\victory-data\diag-journal.txt.
$journal = 'D:\victory-data\diag-journal.txt'
$env:VICTORY_CACHE_DIR = 'D:\victory-data\cache-essai'
$env:VICTORY_D3D12_DEBUG = '1'
$env:VICTORY_D3D12_DRED = '1'
$env:VICTORY_GPU_TIMING = '1'
try {
    Set-Location 'C:\Users\<utilisateur>\Desktop\claude-projects\VIctory-essai'
    & .\target\diag\victory.exe run 'D:\GTA6.Xbox.v1.0.2.1\game_scarlett_master.exe' --execute 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath $journal
} finally {
    $env:VICTORY_CACHE_DIR = ''
    $env:VICTORY_D3D12_DEBUG = ''
    $env:VICTORY_D3D12_DRED = ''
    $env:VICTORY_GPU_TIMING = ''
    Set-Location 'C:\Users\<utilisateur>\Desktop\claude-projects\VIctory'
}
