# Vérifie la correction des dérivées implicites en calcul : tests du traducteur, nouvelle traduction,
# puis préparation des pipelines sans liste de refus (le compilateur AMD plante-t-il encore ?).
$g = 'D:\GTA6.Xbox.v1.0.2.1'
$fichiers = "$g\shaders.rpf", "$g\game_scarlett_master.exe", "$g\XboxBvhBuilder.dll", "$g\amd_fidelityfx_upscaler_xbox.dll"
Set-Location 'D:\victory-src\nouveau'
$t = @()
$t | Set-Content D:\victory-data\tests-shader.txt
$ok = ($t | Select-String 'test result: ok\. (\d+) passed' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Measure-Object -Sum).Sum
$ko = ($t | Select-String 'test result: \w+\. \d+ passed; (\d+) failed' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Measure-Object -Sum).Sum
"Tests du traducteur : $ok réussis, $ko en échec"
$t | Select-String '^test .* FAILED|panicked at' | Select-Object -First 5 | ForEach-Object { "  $($_.Line)" }
$t0 = Get-Date
.\target\diag\victory.exe translate --cache @fichiers *> D:\victory-data\translate-cache-2.txt
"Traduction : code $LASTEXITCODE en $([int]((Get-Date) - $t0).TotalSeconds) s ; $((Get-Content D:\victory-data\translate-cache-2.txt -Tail 1))"
$env:VICTORY_REFUSE_SHADERS = ''
$env:VICTORY_PIPELINE_LOG = 'D:\victory-data\pipelines-log-2.txt'
Remove-Item $env:VICTORY_PIPELINE_LOG -ErrorAction SilentlyContinue
$t0 = Get-Date
.\target\diag\victory.exe translate --cache --pipelines @fichiers *> D:\victory-data\translate-pipelines-5.txt
"Pipelines : code $LASTEXITCODE en $([int]((Get-Date) - $t0).TotalSeconds) s"
Get-Content D:\victory-data\translate-pipelines-5.txt | Select-String 'pipelines to make' | Select-Object -Last 1 | ForEach-Object { "  $($_.Line)" }
$env:VICTORY_PIPELINE_LOG = ''
