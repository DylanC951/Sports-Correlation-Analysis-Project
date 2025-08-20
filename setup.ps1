# Create directory structure
$folders = @(
    "backtesting",
    "backtesting/utils",
    "data",
    "data/historical_stats", 
    "data/historical_odds"
)

foreach ($folder in $folders) {
    if (!(Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder -Force
        Write-Host "Created directory: $folder"
    }
}

# Create empty Python init files
$initFiles = @(
    "backtesting/__init__.py",
    "backtesting/utils/__init__.py"
)

foreach ($file in $initFiles) {
    if (!(Test-Path $file)) {
        New-Item -ItemType File -Path $file
        Write-Host "Created file: $file"
    }
}

Write-Host "Folder structure created successfully"