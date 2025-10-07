#!/bin/bash

# Quick Start Script for Stage 2 Educational Alignment
# This script runs the complete Stage 2 workflow

set -e

echo "🚀 Quick Start: Stage 2 Educational Alignment"
echo "=============================================="
echo ""

# Check if Stage 1 is complete
if [ ! -d "stage1_output" ]; then
    echo "❌ Stage 1 output not found!"
    echo "Please run Stage 1 training first:"
    echo "  ./quick_start_stage1.sh"
    exit 1
fi

# Check if Stage 1 data exists
if [ ! -d "stage1_data" ]; then
    echo "❌ Stage 1 data not found!"
    echo "Please run Stage 1 data preparation first:"
    echo "  python stage1_data_preparation.py"
    exit 1
fi

echo "✅ Stage 1 data found"
echo ""

# Step 1: Data Preparation
echo "📚 Step 1: Preparing Stage 2 Educational Dataset"
echo "================================================"
python stage2_data_preparation.py

if [ $? -ne 0 ]; then
    echo "❌ Stage 2 data preparation failed!"
    exit 1
fi

echo "✅ Stage 2 data preparation complete"
echo ""

# Step 2: Training
echo "🏋️ Step 2: Training Stage 2 Model"
echo "=================================="
./run_stage2_training.sh

if [ $? -ne 0 ]; then
    echo "❌ Stage 2 training failed!"
    exit 1
fi

echo "✅ Stage 2 training complete"
echo ""

# Step 3: Evaluation
echo "📊 Step 3: Evaluating Stage 2 Model"
echo "==================================="
python stage2_evaluation.py

if [ $? -ne 0 ]; then
    echo "❌ Stage 2 evaluation failed!"
    exit 1
fi

echo "✅ Stage 2 evaluation complete"
echo ""

# Summary
echo "🎉 Stage 2 Educational Alignment Complete!"
echo "=========================================="
echo ""
echo "📁 Output Files:"
echo "  - stage2_data/          # Educational dataset"
echo "  - stage2_output/        # Trained model"
echo "  - stage2_evaluation/    # Evaluation results"
echo ""
echo "📊 Key Features:"
echo "  - Comprehensive educational responses"
echo "  - Diagnosis, symptoms, precautions, education, questions"
echo "  - Medical safety disclaimers"
echo "  - Professional-quality guidance"
echo ""
echo "🚀 Next Steps:"
echo "  1. Review evaluation results in stage2_evaluation/"
echo "  2. Test the model with sample images"
echo "  3. Deploy for real-world testing"
echo "  4. Collect user feedback for improvements"
echo ""
echo "The model is now ready to provide comprehensive, educational,"
echo "and safe dermatological guidance!"
