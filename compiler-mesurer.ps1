# Compile la version combinée (main + #1632 + #1418), lance les tests, puis mesure la traduction des shaders.
$ErrorActionPreference = 'Continue'
$wt  = 'C:\Users\<utilisateur>\Desktop\claude-projects\VIctory-essai'
$out = 'C:\Users\<utilisateur>\AppData\Local\Temp\claude\589da39571b35af2\eace1f65-7086-4846-ba0e-fec6603af29e\scratchpad\essai'
$jeu = 'D:\GTA6.Xbox.v1.0.2.1'
$res = Join-Path $out 'resume.txt'
Set-Location $wt
"Début : $(Get-Date -Format 'HH:mm:ss')" | Set-Content $res
cargo build --release --quiet -p victory 2>&1 | Out-File (Join-Path $out 'build.txt')
if ($LASTEXITCODE -ne 0) { "ÉCHEC de compilation (voir build.txt)" | Add-Content $res; exit 1 }
"Compilation réussie : $(Get-Date -Format 'HH:mm:ss')" | Add-Content $res
$t = cargo test --release --quiet --workspace 2>&1
$t | Out-File (Join-Path $out 'tests.txt')
$ok = ($t | Select-String 'test result: ok\. (\d+) passed' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Measure-Object -Sum).Sum
$ko = ($t | Select-String '(\d+) failed' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Measure-Object -Sum).Sum
"Tests : $ok réussis, $ko en échec" | Add-Content $res
.\target\release\victory.exe translate --validate --pipelines --list $jeu *> (Join-Path $out 'translate.txt')
(Get-Content (Join-Path $out 'translate.txt')) | Where-Object { $_ -match '^DXIL |^\s+\d+ emitted of' } | ForEach-Object { $_.Trim() } | Add-Content $res
"Fin : $(Get-Date -Format 'HH:mm:ss')" | Add-Content $res
