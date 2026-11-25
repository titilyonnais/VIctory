# Lance le nouveau main (D:\victory-src\nouveau, version de diagnostic) avec DRED, sans la couche de débogage.
# Cache : celui de victory.toml (D:\victory-data\cache-nouveau). Journal : D:\victory-data\nouveau.txt.
param([string]$Journal = 'D:\victory-data\nouveau.txt')
$env:VICTORY_CACHE_DIR = ''
$env:VICTORY_D3D12_DRED = '1'
try {
    Set-Location 'D:\victory-src\nouveau'
    & .\target\diag\victory.exe run 'D:\GTA6.Xbox.v1.0.2.1\game_scarlett_master.exe' --execute 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath $Journal
} finally {
    $env:VICTORY_D3D12_DRED = ''
    Set-Location 'C:\Users\<utilisateur>\Desktop\claude-projects\VIctory'
}
