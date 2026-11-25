# Résume un journal de diagnostic : durée, fin, soumissions lentes, dessins lents par shader de maillage.
param([string]$Journal)
$l = Get-Content $Journal | ForEach-Object { $_ -replace "\x1b\[[0-9;]*m", '' }
$ts = $l | Where-Object { $_ -match '^\d{4}-\d\d-\d\dT' }
$t0 = [datetime]($ts[0].Substring(0, 27)); $t1 = [datetime]($ts[-1].Substring(0, 27))
"Journal : $Journal ($($l.Count) lignes), durée $([int]($t1 - $t0).TotalSeconds) s"
$f = $l | Select-String 'frame=(\d+)' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Measure-Object -Maximum
"Dernière image vue : $($f.Maximum)"
"Fin : " + (($l | Select-String 'unhandled exception|device removed|panicked|BacktraceBreakpad' | Select-Object -First 1).Line -replace '^(.{0,200}).*', '$1')
$ms = $l | Select-String 'slow host submission ms=(\d+)' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value }
"Soumissions lentes : $($ms.Count), max $(($ms | Measure-Object -Maximum).Maximum) ms, > 2000 ms : $(@($ms | Where-Object { $_ -gt 2000 }).Count)"
$fps = $l | Select-String 'victory_gpu::host: ([\d.]+) fps, ([\d.]+) ms/frame' | Select-Object -Last 3 | ForEach-Object { $_.Matches[0].Value -replace '.*host: ', '' }
"Derniers relevés : $($fps -join ' | ')"
$l | Select-String 'slow list command, ([\d.]+) ms: \w+ \d+ \[mesh ([0-9a-f]+)' | ForEach-Object {
    $h = $_.Matches[0].Groups[2].Value
    [pscustomobject]@{ ms = [double]$_.Matches[0].Groups[1].Value; h = $h; null = $_.Line -match "\| ${h}: [^|]*: null\]" }
} | Group-Object h | ForEach-Object {
    [pscustomobject]@{ maillage = $_.Name; lents = $_.Count; total_ms = [int](($_.Group | Measure-Object ms -Sum).Sum); max_ms = ($_.Group | Measure-Object ms -Maximum).Maximum; avec_null = @($_.Group | Where-Object null).Count }
} | Sort-Object total_ms -Descending | Select-Object -First 6 | Format-Table -AutoSize | Out-String -Width 120
"Messages D3D12 : $(@($l | Select-String 'D3D12: ').Count) ; avertissements uniques :"
$l | Select-String ' WARN ' | Where-Object { $_.Line -notmatch 'slow |D3D12: ' } | ForEach-Object { ($_.Line -replace '^\S+\s+WARN\s+', '') -replace '^(.{0,110}).*', '$1' } | Group-Object | Sort-Object Count -Descending | Select-Object -First 8 | ForEach-Object { "  $($_.Count) × $($_.Name)" }
