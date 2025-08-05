import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Azure OpenAI Configuration
    AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
    AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1")
    AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    
    # Authentication Configuration
    APP_USERNAME = os.getenv("APP_USERNAME")
    APP_PASSWORD = os.getenv("APP_PASSWORD")

    # Paths
    DATA_DIR = "./data"
    MATERIAL_DIR = os.path.join(DATA_DIR, "material")
    QUESTIONS_DIR = os.path.join(DATA_DIR, "questions")
    VECTOR_DB_DIR = "./vector_db"
    
    # RAG Parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 5  # Number of documents to retrieve
    
    # LLM Parameters
    TEMPERATURE = 0.1
    
    # App Settings
    APP_TITLE = "ML Exam Prep Assistant"
    APP_DESCRIPTION = """
    Asistent za pripremu ispita iz mašinskog učenja.
    Postavite pitanja o materijalu, generirajte kvizove ili vežbajte sa ispitnim pitanjima.
    """
    
    @staticmethod
    def validate_config():
        """Validate that required configuration is present"""
        if not Config.AZURE_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT nije podešen. Dodajte ga u .env fajl.")
        if not Config.AZURE_API_KEY:
            raise ValueError("AZURE_OPENAI_API_KEY nije podešen. Dodajte ga u .env fajl.")
        if not Config.APP_USERNAME:
            raise ValueError("APP_USERNAME nije podešen. Dodajte ga u .env fajl.")
        if not Config.APP_PASSWORD:
            raise ValueError("APP_PASSWORD nije podešen. Dodajte ga u .env fajl.")
        
        # Create directories if they don't exist
        os.makedirs(Config.MATERIAL_DIR, exist_ok=True)
        os.makedirs(Config.QUESTIONS_DIR, exist_ok=True)
        os.makedirs(Config.VECTOR_DB_DIR, exist_ok=True)
        
        return True
