# Run this in a NEW terminal while the server is running:
#   .\test_api.ps1
$base = "http://127.0.0.1:8000"

Write-Host "`n1) Health check"
Invoke-RestMethod "$base/health"

Write-Host "`n2) Train model"
Invoke-RestMethod -Method Post -Uri "$base/train"

Write-Host "`n3) Predict (good student)"
Invoke-RestMethod -Method Post -Uri "$base/predict" -ContentType "application/json" -Body '{"hours_studied": 8, "attendance": 90}'

Write-Host "`n4) Predict (weak student)"
Invoke-RestMethod -Method Post -Uri "$base/predict" -ContentType "application/json" -Body '{"hours_studied": 1, "attendance": 40}'
