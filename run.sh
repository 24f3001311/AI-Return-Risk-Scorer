#!/bin/bash
# ============================================
# AI Return-Risk Scorer — One-Click Launch
# Linux/macOS deployment script
# ============================================

set -e

echo "[+] AI Return-Risk Scorer — Starting Up"
echo "========================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker is not installed. Please install Docker first.${NC}"
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}[ERROR] Docker Compose is not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] Docker found${NC}"

# Check if model exists
if [ ! -f "src/api/models/random_forest.pkl" ]; then
    echo -e "${YELLOW}[WARNING] Model not found. Training the model first...${NC}"
    
    # Install Python dependencies
    pip install -r requirements.txt
    
    # Generate data
    echo "[*] Generating synthetic data..."
    python -m src.ml_pipeline.generate_data
    
    # Feature engineering
    echo "[*] Running feature engineering..."
    python -m src.ml_pipeline.feature_engineer
    
    # Train model
    echo "[*] Training Random Forest model..."
    python -m src.ml_pipeline.train_model
    
    echo -e "${GREEN}[OK] Model training complete!${NC}"
fi

# Build and start containers
echo ""
echo "[*] Building and starting Docker containers..."

if docker compose version &> /dev/null 2>&1; then
    docker compose up --build -d
else
    docker-compose up --build -d
fi

echo ""
echo "========================================"
echo -e "${GREEN}[OK] AI Return-Risk Scorer is running!${NC}"
echo ""
echo "   Backend API:    http://localhost:8000"
echo "   API Docs:       http://localhost:8000/docs"
echo "   Frontend:       http://localhost:5173"
echo "   Health Check:   http://localhost:8000/api/health"
echo ""
echo "To stop: docker compose down"
echo "========================================"
