$ErrorActionPreference = "Stop"
$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = Join-Path $packageRoot "SHA256SUMS.txt"
$failed = $false

foreach ($line in Get-Content -LiteralPath $manifestPath) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $parts = $line -split "\s+", 2
    $expected = $parts[0].ToLowerInvariant()
    $relativePath = $parts[1].Trim()
    $filePath = Join-Path $packageRoot $relativePath
    if (-not (Test-Path -LiteralPath $filePath)) {
        Write-Host "MISSING  $relativePath" -ForegroundColor Red
        $failed = $true
        continue
    }
    $actual = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -eq $expected) {
        Write-Host "OK       $relativePath" -ForegroundColor Green
    } else {
        Write-Host "MISMATCH $relativePath" -ForegroundColor Red
        $failed = $true
    }
}

if ($failed) { exit 1 }
Write-Host "All frozen checkpoint hashes match." -ForegroundColor Green
