# Lance le nouveau main (D:\victory-src\nouveau, version de diagnostic) avec DRED et la liste des shaders
# qui font planter le compilateur AMD (voir chasser-plantages-amd.ps1). Journal : D:\victory-data\nouveau-refus.txt.
param([string]$Journal = 'D:\victory-data\nouveau-refus.txt')
$env:VICTORY_CACHE_DIR = ''
$env:VICTORY_D3D12_DRED = '1'
$env:VICTORY_REFUSE_SHADERS = (Get-Content 'D:\victory-data\scripts\refus-amd.txt' | Where-Object { $_ -match '^[0-9a-f]{16}$' }) -join ','
try {
    Set-Location 'D:\victory-src\nouveau'
    & .\target\diag\victory.exe run 'D:\GTA6.Xbox.v1.0.2.1\game_scarlett_master.exe' --execute 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath $Journal
} finally {
    $env:VICTORY_D3D12_DRED = ''
    $env:VICTORY_REFUSE_SHADERS = ''
    Set-Location 'C:\Users\<utilisateur>\Desktop\claude-projects\VIctory'
}
