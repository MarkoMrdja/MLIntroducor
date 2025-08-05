# ML Exam Prep Assistant 🤖

An intelligent RAG (Retrieval-Augmented Generation) system designed specifically for preparing for Machine Learning exams in Serbian. Built with the latest LangChain and Azure OpenAI integration.

## 🚀 Features

- **Smart Document Chunking**: Automatically recognizes and optimally chunks different document types:
  - 📚 **Exam Questions**: Split by topics (1., 2., 3., etc.)
  - 📝 **Exercise Sheets**: Split by numbered problems
  - 🎓 **Lecture Slides**: Split by concepts and bullet points
  - 📖 **Practicum/Books**: Semantic chunking for theory

- **Multi-Modal Learning**:
  - 🔍 **Ask Questions**: Get detailed explanations about concepts
  - 🎯 **Generate Quizzes**: Create exam-style questions with multiple choice
  - 📝 **Practice Mode**: Work with actual exam questions and get graded
  - ⚙️ **Smart Assessment**: Detailed feedback on your answers

- **Serbian Language Optimized**: 
  - Uses proper ML terminology in Serbian ("izglednost", "pristrasnost", "varijansa")
  - Exam-style question generation
  - University-level explanations

## 🛠️ Technical Stack

- **LangChain 0.3.27** - Latest version with modern API
- **Azure OpenAI** - Enterprise-grade LLM integration  
- **ChromaDB 1.0.15** - Vector database for semantic search
- **Streamlit 1.47.1** - Interactive web interface
- **Smart Chunking** - Content-aware document processing

## 📋 Prerequisites

- Python 3.11+
- Azure OpenAI account with:
  - GPT-4 deployment (for chat)
  - Text-embedding deployment (for vectors)
- Ubuntu/Linux environment

## ⚡ Quick Setup

### 1. Environment Setup
```bash
# Make sure python3-venv is installed
sudo apt install python3-venv

# Clone/create project
mkdir ml-exam-prep && cd ml-exam-prep

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (will install latest versions)
pip install streamlit langchain openai chromadb pypdf python-dotenv sentence-transformers tiktoken numpy langchain-openai langchain-community
```

### 2. Configuration
Create `.env` file with your Azure OpenAI credentials:
```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_API_VERSION=2024-02-15-preview
AZURE_CHAT_DEPLOYMENT=your-gpt4-deployment-name
AZURE_EMBEDDING_DEPLOYMENT=your-embedding-deployment-name

# Application Authentication (NEW!)
APP_USERNAME=admin
APP_PASSWORD=your-secure-password-here
```

> **🔒 Security Note**: The app now includes basic authentication to prevent unauthorized access when deployed. Default credentials are `admin/changeme123` - **make sure to change these in production!**

### 3. Add Your Materials
```bash
# Copy your study materials
data/
├── material/           # Your PDFs (praktikum, slides, etc.)
│   ├── praktikum.pdf
│   ├── predavanje_*.pdf
│   └── vezbe_*.pdf
└── questions/          # Exam questions
    └── ispitna_pitanja.pdf
```

### 4. Process Documents & Launch
```bash
# Test system
python test_system.py

# Launch application
streamlit run app.py
```

## 📖 How to Use

### 1. **Process Documents**
- Go to "⚙️ Podešavanja" tab
- Click "🔄 Obradi dokumente"
- Wait for processing (~5-7 minutes for full materials)

### 2. **Ask Questions**
```
Example: "Objasni razliku između ML i Bayesove estimacije"
```

### 3. **Generate Quizzes**
- Choose topic and difficulty
- Get exam-style multiple choice questions
- Automatic explanations for correct answers

### 4. **Practice Exam Questions**
- Work with real exam questions from your materials
- Get detailed graded feedback
- Identify areas for improvement

## 🎯 Document Processing Details

### Smart Chunking Strategy

| Document Type | Recognition | Chunking Method | Example |
|---------------|-------------|-----------------|---------|
| Exam Questions | `ispitna_pitanja.pdf` | By topics (1., 2., 3.) | "1. PREGLED LINEARNE ALGEBRE" → 1 chunk |
| Exercises | `vezbe.pdf`, `ponavljanje.pdf` | By problems (1., 2., 3.) | Each numbered problem → 1 chunk |
| Lectures | `predavanje.pdf`, `slajd.pdf` | By concepts (◼, ❑) | Related bullet points → 1 chunk |
| Practicum | `praktikum.pdf` | Semantic (1500 chars) | Complete explanations → 1 chunk |

### Expected Processing Times
- **Exam Questions** (6 pages): ~30 seconds → 15-20 chunks
- **Exercise Sheet** (1 page): ~10 seconds → 15 chunks  
- **Lecture Slides** (17 pages): ~45 seconds → 25-30 chunks
- **Large Practicum** (130 pages): ~3-5 minutes → 150-200 chunks

## 🔧 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Files     │───▶│ Smart Chunking   │───▶│ Vector Database │
│ (Your Material) │    │ (Content-Aware)  │    │   (ChromaDB)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streamlit UI   │◀───│   RAG System     │◀───│ Azure OpenAI    │
│ (Serbian Lang)  │    │ (LangChain 0.3)  │    │ (GPT-4 + Embed) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
# Access at http://localhost:8501
```

### Docker Deployment
```bash
# Build image
docker build -t ml-exam-prep .

# Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/.env:/app/.env ml-exam-prep
```

### Server Deployment
```bash
# On your server
git clone [your-repo]
cd ml-exam-prep

# Copy your data and config
scp -r data/ user@server:/path/to/ml-exam-prep/
scp .env user@server:/path/to/ml-exam-prep/

# Run with docker-compose
docker-compose up -d
```

## 🔍 Testing Your Setup

Run the included test script to verify everything works:
```bash
python test_system.py
```

Expected output:
```
🚀 MLIntroducor System Test
✅ Configuration loaded successfully
✅ Document processor initialized  
✅ RAG system initialized successfully
🎉 All tests passed! System is ready to use.
```

## 📊 Performance & Costs

### Azure OpenAI Usage
- **GPT-4 calls**: ~$0.03 per question/quiz generation
- **Embeddings**: ~$0.0001 per document chunk
- **Total setup cost**: ~$2-5 for full material processing
- **Daily usage**: ~$1-3 for active studying

### System Performance  
- **Query Response**: 2-5 seconds
- **Quiz Generation**: 5-10 seconds
- **Document Processing**: 5-7 minutes (one-time)
- **Memory Usage**: ~500MB RAM

## 🛟 Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'langchain_openai'"**
```bash
pip install langchain-openai langchain-community
```

**2. "Vector store nije inicijalizovan"**
- Go to Settings tab and process documents first
- Check that PDFs are in data/material/ and data/questions/

**3. "Configuration error: AZURE_OPENAI_ENDPOINT nije podešen"**
- Update your .env file with correct Azure credentials
- Ensure API key and endpoint are valid

**4. Slow processing**
- Normal for large documents (130 page practicum takes 3-5 minutes)
- ChromaDB builds indexes which takes time initially

### Debug Mode
```bash
# Run with verbose logging
PYTHONPATH=/path/to/project python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from test_system import main
main()
"
```

## 📚 Example Queries

### Serbian ML Questions:
```
- "Objasni razliku između supervised i unsupervised learning-a"
- "Kako se računa maksimalna izglednost za Gaussovu raspodelu?"
- "Koja je razlika između pristrasnosti i varijanse procene?"
- "Generiši kviz pitanje o neural networks"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎓 Academic Use

This tool is designed for educational purposes to help students prepare for Machine Learning exams. It uses your own course materials and doesn't replace studying the source material - it enhances your learning process with intelligent question-answering and quiz generation.

---

**Built with ❤️ for ML students at Serbian universities**

For support, create an issue on GitHub or contact the maintainers.
