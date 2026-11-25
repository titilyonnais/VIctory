# Le processus du jeu (victory.exe lancé avec « run »), à distinguer des traductions (victory.exe translate)
# qui peuvent tourner en même temps, ici ou dans les sessions parallèles.
param([switch]$Arreter)
$p = Get-CimInstance Win32_Process -Filter "Name='victory.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match '\srun\s' }
if ($Arreter) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
$p | ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue }
