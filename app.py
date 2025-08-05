import streamlit as st
import os
from pathlib import Path
from config import Config
from document_processor import DocumentProcessor
from rag_system import MLExamRAG
from azure_blob_loader import IncrementalDocumentProcessor

# Page configuration
st.set_page_config(
    page_title=Config.APP_TITLE,
    page_icon="🤖",
    layout="wide"
)

def check_authentication():
    """Check if user is authenticated"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Prijavite se")
        st.markdown("Molimo unesite pristupne podatke za korišćenje aplikacije.")
        
        with st.form("login_form"):
            username = st.text_input("Korisničko ime")
            password = st.text_input("Lozinka", type="password")
            submit_button = st.form_submit_button("Prijavite se")
            
            if submit_button:
                if username == Config.APP_USERNAME and password == Config.APP_PASSWORD:
                    st.session_state.authenticated = True
                    st.success("Uspešno ste se prijavili!")
                    st.rerun()
                else:
                    st.error("Neispravno korisničko ime ili lozinka.")
        
        st.markdown("---")
        st.markdown("**Napomena:** Ovo je zaštićena aplikacija. Kontaktirajte administratora za pristupne podatke.")
        return False
    
    return True

def show_logout_option():
    """Show logout option in sidebar"""
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Odjavite se"):
        st.session_state.authenticated = False
        st.rerun()

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached for performance)"""
    try:
        Config.validate_config()
        rag = MLExamRAG(
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            api_version=Config.AZURE_API_VERSION,
            chat_deployment=Config.AZURE_CHAT_DEPLOYMENT,
            embedding_deployment=Config.AZURE_EMBEDDING_DEPLOYMENT
        )
        
        # Try to load existing vector store
        if not rag.load_existing_vectorstore():
            st.warning("Vector store nije pronađen. Molimo učitajte dokumente.")
            return None
        
        return rag
    except Exception as e:
        st.error(f"Greška pri inicijalizaciji: {str(e)}")
        return None

def process_documents_incremental():
    """Process only new or updated documents from Azure Blob Storage"""
    try:
        processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Initialize incremental processor
        incremental_processor = IncrementalDocumentProcessor(
            document_processor=processor,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Process incremental updates
        new_documents, stats = incremental_processor.process_incremental_update()
        
        if stats['new_or_updated'] == 0 and stats['removed'] == 0:
            st.info("Nema novih dokumenata za obradu. Svi dokumenti su već obrađeni.")
            return True
        
        if new_documents:
            # Initialize RAG system
            rag = MLExamRAG(
                azure_endpoint=Config.AZURE_ENDPOINT,
                api_key=Config.AZURE_API_KEY,
                api_version=Config.AZURE_API_VERSION,
                chat_deployment=Config.AZURE_CHAT_DEPLOYMENT,
                embedding_deployment=Config.AZURE_EMBEDDING_DEPLOYMENT
            )
            
            # Add new documents to existing vector store
            rag.add_documents_to_vectorstore(new_documents)
        
        # Handle removed documents
        if stats['removed_doc_ids']:
            # Note: ChromaDB doesn't have easy document removal by metadata
            # For now, we'll track this in the metadata and suggest full rebuild if needed
            st.warning(f"Detektovano je {stats['removed']} uklonjenih dokumenata. "
                      "Za potpuno uklanjanje preporučuje se potpuna obnova baze.")
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Novi/ažurirani dokumenti", stats['new_or_updated'])
        with col2:
            st.metric("Uspešno obrađeni", stats['processed_successfully'])
        with col3:
            st.metric("Uklonjeni dokumenti", stats['removed'])
        
        if stats['processing_errors']:
            st.error("Greške pri obradi:")
            for error in stats['processing_errors']:
                st.error(f"• {error}")
        
        if stats['processed_successfully'] > 0:
            st.success(f"Uspešno obrađeno {stats['processed_successfully']} dokumenata!")
        
        return True
        
    except Exception as e:
        st.error(f"Greška pri obradi dokumenata: {str(e)}")
        return False

def process_documents_full_rebuild():
    """Process all documents from Azure Blob Storage (full rebuild)"""
    try:
        processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Initialize incremental processor
        incremental_processor = IncrementalDocumentProcessor(
            document_processor=processor,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Process all documents
        all_documents, stats = incremental_processor.process_full_rebuild()
        
        if not all_documents:
            st.error("Nisu pronađeni dokumenti u Azure Blob Storage.")
            return False
        
        # Initialize RAG system and rebuild vector store
        rag = MLExamRAG(
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            api_version=Config.AZURE_API_VERSION,
            chat_deployment=Config.AZURE_CHAT_DEPLOYMENT,
            embedding_deployment=Config.AZURE_EMBEDDING_DEPLOYMENT
        )
        rag.setup_vectorstore(all_documents)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Ukupno dokumenata", stats['total_documents'])
        with col2:
            st.metric("Uspešno obrađeni", stats['processed_successfully'])
        
        if stats['processing_errors']:
            st.error("Greške pri obradi:")
            for error in stats['processing_errors']:
                st.error(f"• {error}")
        
        st.success(f"Uspešno obrađeno {stats['processed_successfully']} dokumenata sa {len(all_documents)} chunk-ova!")
        return True
        
    except Exception as e:
        st.error(f"Greška pri obradi dokumenata: {str(e)}")
        return False

def main():
    # Check authentication first
    if not check_authentication():
        return
    
    st.title(Config.APP_TITLE)
    st.markdown(Config.APP_DESCRIPTION)
    
    # Sidebar for navigation
    st.sidebar.title("Navigacija")
    mode = st.sidebar.selectbox(
        "Izaberite režim:",
        ["📚 Postavi pitanje", "🎯 Generiši kviz", "📝 Vežbaj ispitna pitanja", "🧪 Evaluacija sistema", "⚙️ Podešavanja"]
    )
    
    # Show logout option
    show_logout_option()
    
    # Initialize RAG system
    rag = initialize_rag_system()
    
    if mode == "⚙️ Podešavanja":
        st.header("Podešavanja sistema")
        
        # Azure Blob Storage status
        st.subheader("Azure Blob Storage")
        try:
            from azure_blob_loader import IncrementalDocumentProcessor
            processor = DocumentProcessor(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
            incremental_processor = IncrementalDocumentProcessor(processor)
            
            storage_stats = incremental_processor.get_storage_stats()
            
            # Show available documents in storage
            st.write("**Dostupni dokumenti u Azure Blob Storage:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ukupno dostupno", storage_stats['total_available'])
            with col2:
                st.metric("Materijali", storage_stats['material_available'])
            with col3:
                st.metric("Pitanja", storage_stats['question_available'])
            
            # Show processed documents
            if storage_stats['total_documents'] > 0:
                st.write("**Obrađeni dokumenti:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Ukupno obrađeno", storage_stats['total_documents'])
                with col2:
                    st.metric("Chunk-ovi", storage_stats['total_chunks'])
                with col3:
                    if storage_stats['last_update'] != 'Never':
                        st.success("✅ Obrađeni")
                    else:
                        st.warning("⏳ Nisu obrađeni")
                        
                if storage_stats['last_update'] != 'Never':
                    st.info(f"Poslednje ažuriranje: {storage_stats['last_update']}")
            else:
                st.warning("📋 Nijedan dokument nije obrađen. Pokrenite obradu dokumenata da biste kreirali vector store.")
            
            # Debug button
            if st.button("🔍 Debug Blob Connection"):
                st.write("**Testing blob connection...**")
                
                from azure_blob_loader import AzureBlobDocumentLoader
                loader = AzureBlobDocumentLoader(
                    Config.AZURE_STORAGE_CONNECTION_STRING,
                    Config.AZURE_STORAGE_CONTAINER_NAME
                )
                
                # Test connection
                container_client = loader.container_client
                st.write(f"✅ Connected to container: {Config.AZURE_STORAGE_CONTAINER_NAME}")
                
                # List all blobs
                st.write("**All blobs in container:**")
                all_blobs = list(container_client.list_blobs())
                st.write(f"Total blobs found: {len(all_blobs)}")
                
                for blob in all_blobs[:10]:  # Show first 10
                    st.write(f"- {blob.name} (size: {blob.size} bytes)")
                
                # Test specific prefixes
                st.write("**Material blobs:**")
                material_blobs = loader.list_blobs_by_prefix(Config.BLOB_MATERIAL_PREFIX)
                st.write(f"Found {len(material_blobs)} material documents")
                for blob in material_blobs[:5]:
                    st.json(blob)
                
                st.write("**Question blobs:**")
                question_blobs = loader.list_blobs_by_prefix(Config.BLOB_QUESTIONS_PREFIX)
                st.write(f"Found {len(question_blobs)} question documents")
                for blob in question_blobs[:5]:
                    st.json(blob)
                    
        except Exception as e:
            st.error(f"Greška pri povezivanju sa Azure Blob Storage: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        
        # Document processing options
        st.subheader("Obrada dokumenata")
        
        processing_mode = st.radio(
            "Izaberite način obrade:",
            ["🔄 Inkrementalno ažuriranje", "🔥 Potpuna obnova"],
            help="Inkrementalno: obrađuje samo nove/izmenjene dokumente. Potpuna obnova: briše sve i obrađuje ponovo."
        )
        
        if processing_mode == "🔄 Inkrementalno ažuriranje":
            if st.button("🔄 Obradi nove dokumente", type="primary"):
                with st.spinner("Obrađujem nove dokumente..."):
                    if process_documents_incremental():
                        st.rerun()
                        
            st.info("💡 Ova opcija će obrađivati samo nove ili izmenjene dokumente iz Azure Blob Storage.")
            
        else:  # Full rebuild
            st.warning("⚠️ Potpuna obnova će obrisati postojeći vector store i obraditi sve dokumente ponovo.")
            if st.button("� Potpuna obnova", type="secondary"):
                with st.spinner("Obrađujem sve dokumente..."):
                    if process_documents_full_rebuild():
                        st.rerun()
        
        # Instructions
        st.subheader("Instrukcije za postavljanje")
        st.markdown(f"""
        1. **Azure Blob Storage**: Postavite dokumente u container `{Config.AZURE_STORAGE_CONTAINER_NAME}`
           - Materijali: `{Config.BLOB_MATERIAL_PREFIX}` (npr. `material/predavanje_01.pdf`)
           - Pitanja: `{Config.BLOB_QUESTIONS_PREFIX}` (npr. `questions/ispitna_pitanja.pdf`)
        2. **Obradi dokumente**: Kliknite dugme iznad da kreirate vector store
        3. **Azure API**: Uverite se da su Azure OpenAI podaci podešeni u .env fajlu
        4. **Storage**: Dodajte AZURE_STORAGE_CONNECTION_STRING u .env fajl
        """)
    
    elif rag is None:
        st.error("RAG sistem nije inicijalizovan. Idite na Podešavanja da učitate dokumente.")
        return
    
    elif mode == "📚 Postavi pitanje":
        st.header("Postavi pitanje o materijalu")
        
        question = st.text_area(
            "Vaše pitanje:",
            placeholder="Na primer: Objasni razliku između supervised i unsupervised learning-a",
            height=100
        )
        
        if st.button("Pošalji pitanje", type="primary"):
            if question:
                with st.spinner("Tražim odgovor..."):
                    try:
                        answer = rag.ask_question(question, mode="explanation")
                        st.markdown("### Odgovor:")
                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"Greška: {str(e)}")
            else:
                st.warning("Molimo unesite pitanje.")
    
    elif mode == "🎯 Generiši kviz":
        st.header("Generiši kviz pitanja")
        
        col1, col2 = st.columns(2)
        
        with col1:
            topic = st.text_input(
                "Tema (opciono):",
                placeholder="neural networks, SVM, clustering..."
            )
        
        with col2:
            difficulty = st.selectbox(
                "Težina:",
                ["easy", "medium", "hard"],
                index=1
            )
        
        num_questions = st.slider("Broj pitanja:", 1, 5, 1)
        
        if st.button("Generiši kviz", type="primary"):
            with st.spinner("Generiram pitanja..."):
                try:
                    questions = rag.generate_quiz(
                        topic=topic,
                        difficulty=difficulty,
                        num_questions=num_questions
                    )
                    
                    for i, question in enumerate(questions, 1):
                        st.markdown(f"### Pitanje {i}")
                        st.markdown(question)
                        st.markdown("---")
                        
                except Exception as e:
                    st.error(f"Greška: {str(e)}")
    
    elif mode == "📝 Vežbaj ispitna pitanja":
        st.header("Vežbaj sa ispitnim pitanjima")
        
        # Get available exam questions
        try:
            exam_questions = rag.get_exam_questions()
            
            if not exam_questions:
                st.warning("Nema dostupnih ispitnih pitanja. Dodajte ih u data/questions/ direktorijum.")
                return
            
            st.write(f"Dostupno je {len(exam_questions)} ispitnih pitanja.")
            
            # Question selection
            question_choice = st.selectbox(
                "Izaberite pitanje:",
                ["Nasumično pitanje"] + [f"Pitanje {i+1}" for i in range(len(exam_questions))]
            )
            
            if st.button("Prikaži pitanje", type="primary"):
                if question_choice == "Nasumično pitanje":
                    question_data = rag.practice_exam_question()
                else:
                    question_id = int(question_choice.split()[-1]) - 1
                    question_data = rag.practice_exam_question(question_id)
                
                if "error" not in question_data:
                    st.session_state['current_question'] = question_data['question']
            
            # Display current question and answer input
            if 'current_question' in st.session_state:
                st.markdown("### Pitanje:")
                st.markdown(st.session_state['current_question'])
                
                student_answer = st.text_area(
                    "Vaš odgovor:",
                    height=150,
                    key="student_answer"
                )
                
                if st.button("Oceni odgovor"):
                    if student_answer:
                        with st.spinner("Ocenjujem odgovor..."):
                            try:
                                assessment = rag.assess_answer(
                                    st.session_state['current_question'],
                                    student_answer
                                )
                                st.markdown("### Ocena:")
                                st.markdown(assessment)
                            except Exception as e:
                                st.error(f"Greška: {str(e)}")
                    else:
                        st.warning("Molimo unesite odgovor.")
                        
        except Exception as e:
            st.error(f"Greška: {str(e)}")
    
    elif mode == "🧪 Evaluacija sistema":
        from rag_evaluator import create_evaluation_dashboard
        create_evaluation_dashboard()

if __name__ == "__main__":
    main()
