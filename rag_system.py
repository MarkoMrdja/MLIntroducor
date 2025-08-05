import os
import time
from typing import List, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import random

class MLExamRAGWithRetry:
    def __init__(self, azure_endpoint: str, api_key: str, api_version: str = "2024-02-15-preview", 
                 chat_deployment: str = "gpt-4", embedding_deployment: str = "text-embedding-3-small",
                 persist_directory: str = "./vector_db", max_retries: int = 10, retry_delay: int = 3):
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.persist_directory = persist_directory
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize embeddings and LLM with Azure (Updated LangChain API)
        self.embeddings = AzureOpenAIEmbeddingsWithRetry(
            azure_endpoint=azure_endpoint,
            openai_api_key=api_key,
            openai_api_version=api_version,
            azure_deployment=embedding_deployment,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            openai_api_key=api_key,
            openai_api_version=api_version,
            azure_deployment=chat_deployment,
            temperature=0.1
        )
        
        # Initialize vector store
        self.vectorstore = None
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup different prompt templates for different modes"""
        
        # Explanation mode - for understanding concepts
        self.explanation_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Ti si ekspert za mašinsko učenje koji pomaže studentu da se pripremi za ispit iz kursa "Uvod u mašinsko učenje".
Odgovori na pitanje koristeći SAMO informacije iz priloženog materijala.

MATERIJAL IZ KURSA:
{context}

PITANJE STUDENTA:
{question}

INSTRUKCIJE:
- Odgovori jasno i detaljno na srpskom jeziku
- Koristi TAČNO terminologiju iz materijala (npr. "izglednost", "pristrasnost", "varijansa")
- Navedi konkretne formule i definicije iz materijala kad god je moguće
- Objasni korak po korak za složene koncepte
- Ako informacija nije u materijalu, reci "Ova informacija nije pokrivena u datom materijalu"
- Fokusiraj se na praktičnu primenu za ispit

ODGOVOR:
"""
        )
        
        # Quiz generation mode - creates exam-style questions
        self.quiz_prompt = PromptTemplate(
            input_variables=["context", "difficulty"],
            template="""
Na osnovu materijala iz kursa "Uvod u mašinsko učenje", kreiraj ispitno pitanje.

MATERIJAL:
{context}

TEŽINA: {difficulty}

INSTRUKCIJE:
- Kreiraj pitanje sa 4 opcije (A, B, C, D) 
- Samo JEDAN odgovor je tačan
- Pitanje postavi na srpskom jeziku
- Koristi terminologiju iz materijala
- Fokusiraj se na ključne koncepte za ispit
- Stil pitanja treba da odgovara univerzitetskom ispitu

FORMAT ODGOVORA:
PITANJE: [jasno formulisano pitanje]

A) [opcija A]
B) [opcija B] 
C) [opcija C]
D) [opcija D]

TAČAN ODGOVOR: [slovo]

OBJAŠNJENJE: [zašto je tačan odgovor tačan, sa referencom na materijal]
"""
        )
        
        # Assessment mode - for grading student answers
        self.assessment_prompt = PromptTemplate(
            input_variables=["context", "question", "student_answer"],
            template="""
Proceni odgovor studenta na osnovu materijala iz kursa "Uvod u mašinsko učenje".

MATERIJAL IZ KURSA:
{context}

PITANJE:
{question}

ODGOVOR STUDENTA:
{student_answer}

INSTRUKCIJE:
- Oceni odgovor od 1 do 10 (gde je 10 savršen odgovor)
- Poredi sa tačnim informacijama iz materijala
- Proveri da li student koristi ispravnu terminologiju
- Daj konkretne savete za poboljšanje
- Navedi šta student treba da doda ili ispravi

FORMAT ODGOVORA:
OCENA: [1-10]/10

ŠTA JE DOBRO:
- [konkretni pozitivni aspekti odgovora]

OBLASTI ZA POBOLJŠANJE:
- [specifične greške ili nedostaci]

PREPORUKE:
- [konkretni saveti sa referencama na materijal]

ISPRAVKA:
[kako bi trebalo da glasi potpun odgovor]
"""
        )
    
    def setup_vectorstore_with_retry(self, documents: List[Document]):
        """Initialize and populate the vector store with retry logic"""
        print(f"🔄 Creating vector store with {len(documents)} documents...")
        print(f"⚡ Using retry logic: max {self.max_retries} attempts, {self.retry_delay}s delay")
        
        for attempt in range(self.max_retries):
            try:
                print(f"📝 Attempt {attempt + 1}/{self.max_retries} - Creating embeddings...")
                
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                
                print(f"✅ Vector store created successfully with {len(documents)} documents!")
                return True
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    print(f"⚠️  Rate limit hit on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        print(f"😴 Waiting {self.retry_delay} seconds before retry...")
                        time.sleep(self.retry_delay)
                    else:
                        print(f"❌ Max retries ({self.max_retries}) exceeded")
                        raise e
                else:
                    print(f"❌ Non-rate-limit error: {e}")
                    raise e
        
        return False
    
    def setup_vectorstore(self, documents: List[Document]):
        """Wrapper for backward compatibility"""
        return self.setup_vectorstore_with_retry(documents)
    
    def add_documents_to_vectorstore(self, new_documents: List[Document]):
        """Add new documents to existing vector store"""
        if not self.vectorstore:
            # If no existing vectorstore, create one
            return self.setup_vectorstore(new_documents)
        
        # Add documents to existing vectorstore
        try:
            self.vectorstore.add_documents(new_documents)
            print(f"Added {len(new_documents)} new documents to vector store")
            return True
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def load_existing_vectorstore(self):
        """Load existing vector store"""
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("Loaded existing vector store")
            return True
        return False
    
    def ask_question(self, question: str, mode: str = "explanation") -> str:
        """Ask a question to the RAG system"""
        if not self.vectorstore:
            return "Vector store nije inicijalizovan. Molimo učitajte dokumente prvo."
        
        # Retrieve relevant documents
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        if mode == "explanation":
            prompt = self.explanation_prompt.format(context=context, question=question)
        else:
            prompt = f"Kontekst: {context}\n\nPitanje: {question}"
        
        response = self.llm.invoke(prompt)
        return response.content
    
    def generate_quiz(self, topic: str = "", difficulty: str = "medium", num_questions: int = 1) -> List[str]:
        """Generate quiz questions based on the material"""
        if not self.vectorstore:
            return ["Vector store nije inicijalizovan."]
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        questions = []
        
        for _ in range(num_questions):
            # Get random material for quiz generation
            if topic:
                relevant_docs = retriever.invoke(topic)
            else:
                # Get random documents from material (not questions)
                try:
                    # Sample some random content for quiz generation
                    sample_query = "mašinsko učenje algoritam"  # Generic query to get content
                    relevant_docs = retriever.invoke(sample_query)
                except:
                    relevant_docs = []
            
            if relevant_docs:
                context = "\n\n".join([doc.page_content for doc in relevant_docs[:2]])
            else:
                context = "Nema dostupnog materijala"
            
            prompt = self.quiz_prompt.format(context=context, difficulty=difficulty)
            quiz_question = self.llm.invoke(prompt)
            questions.append(quiz_question.content)
        
        return questions
    
    def assess_answer(self, question: str, student_answer: str) -> str:
        """Assess student's answer"""
        if not self.vectorstore:
            return "Vector store nije inicijalizovan."
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        relevant_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = self.assessment_prompt.format(
            context=context,
            question=question,
            student_answer=student_answer
        )
        
        assessment = self.llm.invoke(prompt)
        return assessment.content
    
    def get_exam_questions(self) -> List[str]:
        """Get available exam questions from the database"""
        if not self.vectorstore:
            return []
        
        # Try to get questions by searching for question-related content
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 20}
            )
            results = retriever.invoke("ispitna pitanja tema")
            return [doc.page_content for doc in results if "tema" in doc.page_content.lower()]
        except:
            return []
    
    def practice_exam_question(self, question_id: Optional[int] = None) -> Dict:
        """Get a specific exam question or random one for practice"""
        exam_questions = self.get_exam_questions()
        
        if not exam_questions:
            return {"error": "Nema dostupnih ispitnih pitanja"}
        
        if question_id is not None and question_id < len(exam_questions):
            question = exam_questions[question_id]
        else:
            question = random.choice(exam_questions)
        
        return {
            "question": question,
            "total_questions": len(exam_questions)
        }


class AzureOpenAIEmbeddingsWithRetry(AzureOpenAIEmbeddings):
    """Azure OpenAI Embeddings with built-in retry logic for rate limits"""
    
    def __init__(self, max_retries: int = 10, retry_delay: int = 3, **kwargs):
        # Remove these keys from kwargs if present
        kwargs.pop("max_retries", None)
        kwargs.pop("retry_delay", None)
        super().__init__(**kwargs)
        self.__dict__["max_retries"] = max_retries
        self.__dict__["retry_delay"] = retry_delay
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Override embed_documents with retry logic"""
        return self._embed_with_retry(super().embed_documents, texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Override embed_query with retry logic"""
        return self._embed_with_retry(super().embed_query, text)
    
    def _embed_with_retry(self, embed_func, *args):
        """Generic retry wrapper for embedding functions with batch splitting on rate limit"""
        batch_size = len(args[0]) if isinstance(args[0], list) else 1
        for attempt in range(self.max_retries):
            try:
                return embed_func(*args)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < self.max_retries - 1:
                        print(f"⚠️  Rate limit hit, waiting {self.retry_delay}s... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(self.retry_delay)

                        # If batch size is greater than 1, split the batch and retry
                        if batch_size > 1:
                            print(f"🔄 Splitting batch into smaller chunks to avoid rate limits...")
                            mid_point = batch_size // 2
                            first_half = args[0][:mid_point]
                            second_half = args[0][mid_point:]

                            # Process each half separately
                            return self._embed_with_retry(embed_func, first_half) + self._embed_with_retry(embed_func, second_half)
                    else:
                        print(f"❌ Max retries exceeded for embedding")
                        raise e
                else:
                    raise e

# For backward compatibility
MLExamRAG = MLExamRAGWithRetry
