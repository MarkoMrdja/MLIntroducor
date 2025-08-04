#!/usr/bin/env python3
"""
Simple test script to verify the RAG system is working
"""

import sys
import os
sys.path.append('/home/mrcho/Documents/MLIntroducor')

from config import Config
from document_processor import DocumentProcessor
from rag_system import MLExamRAG

def test_config():
    """Test configuration loading"""
    print("🔧 Testing configuration...")
    try:
        Config.validate_config()
        print("✅ Configuration loaded successfully")
        print(f"   Azure Endpoint: {Config.AZURE_ENDPOINT}")
        print(f"   Chat Deployment: {Config.AZURE_CHAT_DEPLOYMENT}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_document_processor():
    """Test document processing"""
    print("\n📄 Testing document processor...")
    try:
        processor = DocumentProcessor()
        print("✅ Document processor initialized")
        return True
    except Exception as e:
        print(f"❌ Document processor error: {e}")
        return False

def test_rag_initialization():
    """Test RAG system initialization"""
    print("\n🤖 Testing RAG system...")
    try:
        rag = MLExamRAG(
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            api_version=Config.AZURE_API_VERSION,
            chat_deployment=Config.AZURE_CHAT_DEPLOYMENT,
            embedding_deployment=Config.AZURE_EMBEDDING_DEPLOYMENT
        )
        print("✅ RAG system initialized successfully")
        return True
    except Exception as e:
        print(f"❌ RAG system error: {e}")
        return False

def main():
    print("🚀 MLIntroducor System Test\n")
    
    success = True
    success &= test_config()
    success &= test_document_processor()
    success &= test_rag_initialization()
    
    if success:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Copy your PDF files to data/material/ and data/questions/")
        print("2. Run: streamlit run app.py")
    else:
        print("\n❌ Some tests failed. Please check your configuration.")

if __name__ == "__main__":
    main()
