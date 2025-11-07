param(
    [Parameter(Mandatory=$true)]
    [string]$InputUrl,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputUrl
)

Write-Host "===== Football Tracker Stream Worker =====" -ForegroundColor Green
Write-Host "Input: $InputUrl"
if ($OutputUrl) {
    Write-Host "Output: $OutputUrl"
} else {
    Write-Host "Output: DEBUG MODE" -ForegroundColor Yellow
}
Write-Host ""

if ($OutputUrl) {
    python main.py stream --input $InputUrl --output $OutputUrl
} else {
    python main.py stream --input $InputUrl --debug
}
