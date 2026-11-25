# Capture tous les écrans en une image réduite (largeur 1600 px) dans D:\victory-data\captures, sans toucher
# au clavier ni à la souris. Affiche le chemin de l'image. Les captures sont supprimées après analyse.
param([string]$Nom = (Get-Date -Format 'HHmmss'), [int]$Largeur = 1600)
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$dir = 'D:\victory-data\captures'
New-Item -ItemType Directory -Force $dir | Out-Null
$v = [System.Windows.Forms.SystemInformation]::VirtualScreen
$bmp = New-Object System.Drawing.Bitmap $v.Width, $v.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($v.Left, $v.Top, 0, 0, $bmp.Size)
$h = [int]($v.Height * $Largeur / $v.Width)
$small = New-Object System.Drawing.Bitmap $bmp, $Largeur, $h
$path = Join-Path $dir "$Nom.png"
$small.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose(); $small.Dispose()
$path
