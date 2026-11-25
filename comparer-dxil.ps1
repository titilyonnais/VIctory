# Compare les shaders qui font planter le compilateur AMD (refus-amd.txt) avec des shaders de calcul qui passent :
# pour chaque fonctionnalité DXIL (indicateurs « requires » et opérations dx.op.*), la part de chaque groupe qui l'utilise.
$dxc = 'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe'
$dir = 'D:\victory-data\dxil-hote'
$out = 'D:\victory-data\dxil-texte'
New-Item -ItemType Directory -Force $out | Out-Null
$bad = Get-Content D:\victory-data\scripts\refus-amd.txt | Where-Object { $_ -match '^[0-9a-f]{16}$' }
# Shaders de calcul compilés sans problème : ceux des pipelines « built » de la dernière chasse.
$log = Get-Content D:\victory-data\pipelines-log.txt
$builtKeys = $log | Where-Object { $_ -like 'built*' } | ForEach-Object { ($_ -split "`t")[1] }
$good = $log | Where-Object { $_ -like 'start*' -and $_ -match '\[compute ([0-9a-f]{16})\]' -and $builtKeys -contains ($_ -split "`t")[1] } |
    ForEach-Object { [regex]::Match($_, '\[compute ([0-9a-f]{16})\]').Groups[1].Value } | Select-Object -Unique
function Get-Features([string]$h) {
    $f = Join-Path $dir "$h.dxil"
    if (-not (Test-Path $f)) { return $null }
    $txt = Join-Path $out "$h.txt"
    if (-not (Test-Path $txt)) { & $dxc -dumpbin $f -Fc $txt 2>$null | Out-Null }
    $t = Get-Content $txt -ErrorAction SilentlyContinue
    $req = $t | Where-Object { $_ -match '^;\s{2,}(\S.*)$' -and $_ -notmatch 'Name|Index|Mask|----' } | ForEach-Object { 'requires: ' + $_.Trim(' ;') }
    $ops = [regex]::Matches(($t -join "`n"), '@dx\.op\.([A-Za-z0-9]+)') | ForEach-Object { 'op: ' + $_.Groups[1].Value }
    $misc = @()
    if ($t -match 'WaveSize|wavesize') { $misc += 'attr: WaveSize' }
    if ($t -match 'i64 ') { $misc += 'type: i64' }
    if ($t -match 'half |f16') { $misc += 'type: f16' }
    $n = @($t | Where-Object { $_ -match '^\s+%' }).Count
    $misc += "taille: $([int]([math]::Log($n + 1, 2)))"
    @($req + $ops + $misc) | Select-Object -Unique
}
$stats = @{}
foreach ($grp in @(@{n = 'bad'; l = $bad }, @{n = 'good'; l = ($good | Select-Object -First 200) })) {
    $count = 0
    foreach ($h in $grp.l) {
        $fs = Get-Features $h
        if ($null -eq $fs) { continue }
        $count++
        foreach ($x in $fs) { if (-not $stats[$x]) { $stats[$x] = @{ bad = 0; good = 0 } }; $stats[$x][$grp.n]++ }
    }
    Set-Variable -Name "n$($grp.n)" -Value $count
}
"Fautifs analysés : $nbad ; sains analysés : $ngood"
$stats.GetEnumerator() | ForEach-Object {
    [pscustomobject]@{ trait = $_.Key; fautifs = [math]::Round(100 * $_.Value.bad / [math]::Max(1, $nbad)); sains = [math]::Round(100 * $_.Value.good / [math]::Max(1, $ngood)) }
} | Where-Object { $_.fautifs - $_.sains -ge 40 -or $_.sains - $_.fautifs -ge 60 } | Sort-Object { $_.fautifs - $_.sains } -Descending | Format-Table -AutoSize | Out-String -Width 160
