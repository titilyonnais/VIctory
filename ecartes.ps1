# Liste, par leur nom dans le jeu, les shaders dont les dessins sont écartés pendant une partie
# (lignes « dropped N/frame without a host pipeline » des 3 derniers relevés du journal).
param([string]$Journal = 'D:\victory-data\main2.txt', [int]$Top = 40)
$noms = @{}
$effet = ''
foreach ($l in Get-Content D:\victory-data\shader-effects.txt) {
    if ($l -match '^effect (\S+)') { $effet = $Matches[1]; continue }
    foreach ($m in [regex]::Matches($l, '(\w+) (\S+_Wrapped) ([0-9a-f]{16}) [0-9a-f]{16}')) {
        $noms[$m.Groups[3].Value] = "$effet / $($m.Groups[2].Value -replace '_[0-9a-f]{8}_Wrapped$', '')"
    }
}
$j = Get-Content $Journal | ForEach-Object { $_ -replace "\x1b\[[0-9;]*m", '' }
$lignes = $j | Select-String 'dropped ([\d.]+)/frame without a host pipeline: (.*)$'
# Les derniers relevés seulement (le monde).
$derniers = $lignes | Select-Object -Last 120
$derniers | ForEach-Object {
    $n = [double]$_.Matches[0].Groups[1].Value
    $quoi = $_.Matches[0].Groups[2].Value
    $hashes = [regex]::Matches($quoi, '[0-9a-f]{16}') | ForEach-Object { $_.Value }
    [pscustomobject]@{
        par_image = $n
        raison = ($quoi -replace '\[?(compute|vertex|pixel|mesh) [0-9a-f]{16}\]?,? ?', '').Trim(' :')
        shaders = ($hashes | ForEach-Object { if ($noms[$_]) { "$($noms[$_]) ($_)" } else { $_ } }) -join ' + '
    }
} | Sort-Object shaders -Unique | Sort-Object par_image -Descending | Select-Object -First $Top | Format-Table -AutoSize -Wrap | Out-String -Width 220
