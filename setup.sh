#!/bin/bash

echo "🚀 ML Exam Prep Assistant Setup"
echo "================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "❌ Virtual environment not active. Please run:"
    echo "   source venv/bin/activate"
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking installed packages..."

packages=("streamlit" "langchain" "openai" "chromadb" "pypdf")
missing_packages=()

for package in "${packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✅ $package is installed"
    else
        echo "❌ $package is missing"
        missing_packages+=("$package")
    fi
done

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "Installing missing packages..."
    pip install "${missing_packages[@]}"
fi

# Check directories
echo "📁 Checking directory structure..."
mkdir -p data/material data/questions vector_db

if [ -d "data/material" ]; then
    echo "✅ data/material/ directory exists"
    material_count=$(ls data/material/*.pdf 2>/dev/null | wc -l)
    echo "   📄 Found $material_count PDF files"
else
    echo "❌ data/material/ directory missing"
fi

if [ -d "data/questions" ]; then
    echo "✅ data/questions/ directory exists"
    question_count=$(ls data/questions/* 2>/dev/null | wc -l)
    echo "   ❓ Found $question_count question files"
else
    echo "❌ data/questions/ directory missing"
fi

# Check .env file
echo "🔑 Checking configuration..."
if [ -f ".env" ]; then
    echo "✅ .env file exists"
    if grep -q "your-resource-name" .env; then
        echo "⚠️  Please update .env with your actual Azure OpenAI credentials"
    else
        echo "✅ .env appears to be configured"
    fi
else
    echo "❌ .env file missing"
fi

echo ""
echo "🎯 Setup Summary:"
echo "=================="
echo "Next steps:"
echo "1. Update .env file with your Azure OpenAI credentials"
echo "2. Copy your PDF files to data/material/"
echo "3. Copy exam questions to data/questions/"
echo "4. Run: python test_system.py (to verify setup)"
echo "5. Run: streamlit run app.py (to start the application)"
echo ""
echo "Your Azure .env should look like:"
echo "AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/"
echo "AZURE_OPENAI_API_KEY=your_actual_api_key"
echo "AZURE_CHAT_DEPLOYMENT=your_gpt4_deployment_name"
echo "AZURE_EMBEDDING_DEPLOYMENT=your_embedding_model_name"
