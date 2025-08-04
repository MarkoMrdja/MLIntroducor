import os
import json
import re
from typing import List, Dict
from pathlib import Path
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def process_pdf_document(self, pdf_path: str, doc_type: str = "material") -> List[Document]:
        """Process a single PDF and return chunked documents"""
        text = self.extract_text_from_pdf(pdf_path)
        filename = Path(pdf_path).name
        
        # Smart chunking based on document type and content
        chunks = self._smart_chunk_document(text, filename)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "chunk_id": i,
                    "doc_type": doc_type,
                    "total_chunks": len(chunks),
                    "chunk_type": self._identify_chunk_type(chunk, filename)
                }
            )
            documents.append(doc)
        
        return documents
    
    def _smart_chunk_document(self, text: str, filename: str) -> List[str]:
        """Smart chunking based on document structure"""
        
        # For exam questions - split by numbered sections
        if "ispitna pitanja" in filename.lower() or "pitanja" in filename.lower():
            return self._chunk_exam_questions(text)
        
        # For exercise sheets - split by numbered problems
        elif "ponavljanje" in filename.lower() or "vezbe" in filename.lower():
            return self._chunk_exercises(text)
        
        # For lecture slides - split by topics/sections
        elif "predavanje" in filename.lower() or "slajd" in filename.lower():
            return self._chunk_lecture_slides(text)
        
        # For practicum or books - use semantic chunking
        elif "praktikum" in filename.lower():
            return self._chunk_practicum(text)
        
        # Default chunking
        else:
            return self.text_splitter.split_text(text)
    
    def _chunk_exam_questions(self, text: str) -> List[str]:
        """Chunk exam questions by numbered sections"""
        # Split by main sections (1., 2., 3., etc.)
        sections = re.split(r'\n\s*(\d+\.)\s+', text)
        chunks = []
        
        current_chunk = ""
        for i, section in enumerate(sections):
            if re.match(r'\d+\.', section):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = f"Tema {section} "
            else:
                current_chunk += section
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk) > 50]
    
    def _chunk_exercises(self, text: str) -> List[str]:
        """Chunk exercises by numbered problems"""
        # Split by numbered exercises (1., 2., 3., etc.)
        exercises = re.split(r'\n\s*(\d+\.)\s+', text)
        chunks = []
        
        current_chunk = ""
        for i, part in enumerate(exercises):
            if re.match(r'\d+\.', part):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = f"Zadatak {part} "
            else:
                current_chunk += part
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk) > 30]
    
    def _chunk_lecture_slides(self, text: str) -> List[str]:
        """Chunk lecture slides by topics and bullet points"""
        # Split by major headers (◼) and topics
        sections = re.split(r'◼\s+', text)
        chunks = []
        
        for section in sections:
            if len(section.strip()) > 100:
                # Further split long sections by bullet points or paragraphs
                sub_chunks = re.split(r'❑\s+', section)
                
                if len(sub_chunks) > 1:
                    # Combine 2-3 bullet points per chunk
                    for i in range(0, len(sub_chunks), 2):
                        chunk_parts = sub_chunks[i:i+2]
                        combined_chunk = "❑ ".join(chunk_parts).strip()
                        if len(combined_chunk) > 50:
                            chunks.append(combined_chunk)
                else:
                    # Single section, check if too long
                    if len(section) > 800:
                        # Split by sentences/paragraphs
                        sub_parts = self.text_splitter.split_text(section)
                        chunks.extend(sub_parts)
                    else:
                        chunks.append(section.strip())
        
        return [chunk for chunk in chunks if len(chunk) > 50]
    
    def _chunk_practicum(self, text: str) -> List[str]:
        """Chunk practicum/book content by chapters and sections"""
        # Use standard text splitter but with larger chunks for books
        larger_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Larger chunks for book content
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        return larger_splitter.split_text(text)
    
    def _identify_chunk_type(self, chunk: str, filename: str) -> str:
        """Identify the type of chunk for better retrieval"""
        
        if "zadatak" in chunk.lower() or re.search(r'\d+\.', chunk[:20]):
            return "exercise"
        elif "tema" in chunk.lower() or "ispitna pitanja" in filename.lower():
            return "exam_question"
        elif "◼" in chunk or "❑" in chunk:
            return "lecture_content"
        elif "praktikum" in filename.lower():
            return "theory"
        else:
            return "general"
    
    def process_exam_questions(self, questions_file: str) -> List[Document]:
        """Process exam questions from text/JSON file"""
        documents = []
        
        if questions_file.endswith('.json'):
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
                
            for i, question in enumerate(questions_data):
                doc = Document(
                    page_content=f"Pitanje: {question.get('question', '')}\nOdgovor: {question.get('answer', '')}",
                    metadata={
                        "source": "exam_questions",
                        "question_id": i,
                        "doc_type": "question",
                        "difficulty": question.get('difficulty', 'medium')
                    }
                )
                documents.append(doc)
        else:
            # Plain text file with questions
            with open(questions_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by double newline or question numbers
            questions = content.split('\n\n')
            for i, question in enumerate(questions):
                if question.strip():
                    doc = Document(
                        page_content=question.strip(),
                        metadata={
                            "source": "exam_questions",
                            "question_id": i,
                            "doc_type": "question"
                        }
                    )
                    documents.append(doc)
        
        return documents
    
    def process_all_documents(self, material_dir: str, questions_dir: str = None) -> List[Document]:
        """Process all documents in the material directory"""
        all_documents = []
        
        # Process material PDFs
        material_path = Path(material_dir)
        for pdf_file in material_path.glob("*.pdf"):
            print(f"Processing material: {pdf_file.name}")
            docs = self.process_pdf_document(str(pdf_file), "material")
            all_documents.extend(docs)
        
        # Process exam questions if provided
        if questions_dir:
            questions_path = Path(questions_dir)
            for file in questions_path.glob("*"):
                if file.suffix in ['.txt', '.json']:
                    print(f"Processing questions: {file.name}")
                    docs = self.process_exam_questions(str(file))
                    all_documents.extend(docs)
                elif file.suffix == '.pdf':
                    print(f"Processing question PDF: {file.name}")
                    docs = self.process_pdf_document(str(file), "question")
                    all_documents.extend(docs)
        
        print(f"Total documents processed: {len(all_documents)}")
        return all_documents
    def _smart_chunk_document(self, text: str, filename: str) -> List[str]:
        """Smart chunking based on document structure"""
        filename_lower = filename.lower()
        
        # For exam questions - split by numbered sections
        if any(keyword in filename_lower for keyword in ['ispitna', 'pitanja', 'ispit', 'questions']):
            return self._chunk_exam_questions(text)
        
        # For exercise sheets - split by numbered problems  
        elif any(keyword in filename_lower for keyword in ['vezbe', 'ponavljanje', 'zadaci', 'exercises', 'homework']):
            return self._chunk_exercises(text)
        
        # For lecture slides - split by topics/sections
        elif any(keyword in filename_lower for keyword in ['predavanje', 'slajd', 'prezentacija', 'lecture', 'slides']):
            return self._chunk_lecture_slides(text)
        
        # For practicum or books - use semantic chunking
        elif any(keyword in filename_lower for keyword in ['praktikum', 'knjiga', 'skripta', 'teorija', 'textbook']):
            return self._chunk_practicum(text)
        
        # Content-based detection fallback
        else:
            return self._detect_content_type_and_chunk(text, filename)
    
    def _detect_content_type_and_chunk(self, text: str, filename: str) -> List[str]:
        """Fallback: detect document type based on content patterns"""
        
        # Check for exam question patterns
        if re.search(r'\n\s*\d+\.\s+[A-ZŠĐČĆŽ][A-Za-zšđčćžŠĐČĆŽ\s]+', text):
            print(f"🔍 Detected exam questions in {filename} (by content)")
            return self._chunk_exam_questions(text)
        
        # Check for exercise patterns (numbered problems)
        elif re.search(r'\n\s*\d+\.\s*[A-Za-z][^.]*\.', text):
            print(f"🔍 Detected exercises in {filename} (by content)")
            return self._chunk_exercises(text)
        
        # Check for lecture slide patterns (◼, ❑, bullet points)
        elif '◼' in text or '❑' in text or text.count('•') > 10:
            print(f"🔍 Detected lecture slides in {filename} (by content)")
            return self._chunk_lecture_slides(text)
        
        # Default to practicum chunking for long text
        else:
            print(f"🔍 Using default semantic chunking for {filename}")
            return self._chunk_practicum(text)
