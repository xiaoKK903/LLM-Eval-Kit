param(
    [string]$HtmlPath,
    [string]$OutputPath
)

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms

# Open HTML in Edge (WebView2-based or legacy)
Write-Host "Opening HTML report in browser..."
Start-Process "msedge.exe" -ArgumentList "--new-window", "`"file:///$HtmlPath`""
Start-Sleep -Seconds 4

# Minimize all windows to focus on the browser
[System.Windows.Forms.SendKeys]::SendWait("%{Tab}")
Start-Sleep -Seconds 1

# Take screenshot of the entire screen
Write-Host "Taking screenshot..."
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($bounds.X, $bounds.Y, 0, 0, $bounds.Size)
$bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()

Write-Host "Screenshot saved to: $OutputPath"
