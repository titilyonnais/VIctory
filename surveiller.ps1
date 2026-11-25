# Surveille un essai : capture d'écran toutes les $Tous minutes, arrêt du jeu au bout de $Minutes minutes
# (comptées depuis le lancement de ce script). Se termine dès que le jeu s'arrête.
param([string]$Nom = 'essai', [int]$Minutes = 15, [int]$Tous = 5)
$debut = Get-Date
$fin = $debut.AddMinutes($Minutes)
$prochaine = $debut.AddMinutes($Tous)
Start-Sleep -Seconds 20
while ((Get-Date) -lt $fin) {
    if (-not (& 'D:\victory-data\scripts\jeu.ps1')) { "Le jeu s'est arrêté après $([int]((Get-Date) - $debut).TotalMinutes) min"; exit 0 }
    if ((Get-Date) -ge $prochaine) {
        $c = & 'D:\victory-data\scripts\capture-ecran.ps1' -Nom "$Nom-$(Get-Date -Format 'HHmm')"
        "Capture : $c"
        $prochaine = (Get-Date).AddMinutes($Tous)
    }
    Start-Sleep -Seconds 10
}
$c = & 'D:\victory-data\scripts\capture-ecran.ps1' -Nom "$Nom-fin"
"Capture : $c"
& 'D:\victory-data\scripts\jeu.ps1' -Arreter | Out-Null
"Jeu arrêté au bout de $Minutes min"
