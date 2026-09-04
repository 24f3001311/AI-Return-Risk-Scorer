# ============================================
# AI Return-Risk Scorer  One-Click Launch
# Windows PowerShell deployment script
# ============================================

Write-Host " AI Return-Risk Scorer  Starting Up" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check Docker
$dockerExists = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerExists) {
    Write-Host " Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    Write-Host "   Visit: https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor Yellow
    exit 1
}

Write-Host " Docker found" -ForegroundColor Green

# Check if model exists
$modelPath = "src\api\models\random_forest.pkl"
if (-not (Test-Path $modelPath)) {
    Write-Host "  Model not found. Training the model first..." -ForegroundColor Yellow
    
    # Install Python dependencies
    pip install -r requirements.txt
    
    # Generate data
    Write-Host " Generating synthetic data..." -ForegroundColor Cyan
    python -m src.ml_pipeline.generate_data
    
    # Feature engineering
    Write-Host " Running feature engineering..." -ForegroundColor Cyan
    python -m src.ml_pipeline.feature_engineer
    
    # Train model
    Write-Host " Training Random Forest model..." -ForegroundColor Cyan
    python -m src.ml_pipeline.train_model
    
    Write-Host " Model training complete!" -ForegroundColor Green
}

# Build and start containers
Write-Host ""
Write-Host " Building and starting Docker containers..." -ForegroundColor Cyan

docker compose up --build -d

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " AI Return-Risk Scorer is running!" -ForegroundColor Green
Write-Host ""
Write-Host " Backend API:    http://localhost:8000" -ForegroundColor White
Write-Host " API Docs:       http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Frontend:       http://localhost:5173" -ForegroundColor White
Write-Host "  Health Check:   http://localhost:8000/api/health" -ForegroundColor White
Write-Host ""
Write-Host "To stop: docker compose down" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan

