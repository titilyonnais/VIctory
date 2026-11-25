# Scripts d'essai de VIctory (Windows, PowerShell)

Outils utilisés pour tester les piles de correctifs (`essai/pile*`) sur un PC Windows 11 avec une RX 9070 XT.
Aucune donnée du jeu ici : seulement des scripts et des listes d'empreintes de shaders.

- `preparer.ps1 -Src <copie> [-Liste <refus.txt>]` : compile (versions normale et diag), traduit les shaders dans le cache, puis construit les pipelines en chassant les plantages du compilateur AMD (les shaders fautifs vont dans la liste de refus).
- `lancer-auto.ps1 -Src <copie> -Nom <essai>` : lance le jeu avec le diagnostic DRED, les touches automatiques (Entrée pour passer l'écran titre) et des captures d'images ; journal dans `D:\victory-data\<essai>.txt`.
- `surveiller.ps1` : captures d'écran régulières, puis arrêt du jeu au bout de N minutes. `jeu.ps1` : trouve (ou arrête) le processus du jeu seulement (`victory.exe run`), sans toucher aux traductions.
- `fin.ps1 <journal>` : temps passé dans le monde, dernière image, cause de l'arrêt. `ecartes.ps1` : shaders écartés, avec leurs noms.
- `refus-amd.txt` : empreintes des shaders qui faisaient planter amdxc64 (avant le correctif `correctif/amd-tampons-bruts`).

Les chemins (`D:\victory-src`, `D:\victory-data`, dossier du jeu) sont ceux de la machine d'essai : à adapter.
