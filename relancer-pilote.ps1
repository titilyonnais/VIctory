# Dernier recours quand l'affichage reste bloqué après l'arrêt du jeu : envoie le raccourci de Windows
# Win + Ctrl + Maj + B, qui relance le pilote graphique (l'écran clignote, un bip retentit).
Stop-Process -Name victory -Force -ErrorAction SilentlyContinue
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class Touches {
    [DllImport("user32.dll")] public static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
}
'@
$KEYUP = 2
$touches = 0x5B, 0x11, 0x10, 0x42   # Win, Ctrl, Maj, B
foreach ($k in $touches) { [Touches]::keybd_event([byte]$k, 0, 0, [UIntPtr]::Zero) }
Start-Sleep -Milliseconds 100
[array]::Reverse($touches)
foreach ($k in $touches) { [Touches]::keybd_event([byte]$k, 0, $KEYUP, [UIntPtr]::Zero) }
"Raccourci de relance du pilote envoyé à $(Get-Date -Format 'HH:mm:ss')"
