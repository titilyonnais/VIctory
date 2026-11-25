# Résume la fin d'un lancement : temps dans le monde, dernière image, cause de l'arrêt, commande en cours.
param([string]$Journal = 'D:\victory-data\diag-pile.txt')
$l = Get-Content $Journal | ForEach-Object { $_ -replace "\x1b\[[0-9;]*m", '' }
$e = $l | Select-String '^(\S+Z).*frame=(\d+)' | ForEach-Object { [pscustomobject]@{ t = [datetime]$_.Matches[0].Groups[1].Value; f = [int]$_.Matches[0].Groups[2].Value } }
$w = $e | Where-Object { $_.f -ge 1600 } | Select-Object -First 1
$last = [datetime](($l | Where-Object { $_ -match '^\d{4}-' } | Select-Object -Last 1).Substring(0, 27))
"Journal : $Journal"
"Temps dans le monde : $(if ($w) { [int]($last - $w.t).TotalSeconds } else { 'pas atteint' }) s ; dernière image : $(($e | Measure-Object f -Maximum).Maximum)"
$fps = $l | Select-String 'victory_gpu::host: ([\d.]+) fps, ([\d.]+) ms/frame' | Select-Object -Last 2 | ForEach-Object { $_.Matches[0].Value -replace '.*host: ', '' }
"Vitesse à la fin : $($fps -join ' | ')"
$line = ($l | Select-String 'ERROR victory_gpu::host: host GPU device removed' | Select-Object -First 1).Line
if ($line) {
    "Fin : " + (($line -replace '^.*device removed', 'device removed') -replace '; the nearest.*$', '' -replace '^(.{0,900}).*', '$1')
    if ($line -match '(a list stopped at command \d+ of \d+ \(op \d+\))') { $Matches[1] }
    if ($line -match 'list \d+ of \d+, of its \d+: ([^\]]*\])') { "Commande en cours : $($Matches[1])" }
    if ($line -match '(DXGI_ERROR_\w+)') { "Raison : $($Matches[1])" }
} else {
    $x = $l | Select-String 'unhandled exception|panicked|BacktraceBreakpad' | Select-Object -First 1
    "Fin : $(if ($x) { $x.Line -replace '^(.{0,200}).*', '$1' } else { 'fermé sans erreur dans le journal' })"
}
