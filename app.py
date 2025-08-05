import streamlit as st
import os
from pathlib import Path
from config import Config
from document_processor import DocumentProcessor
from rag_system import MLExamRAG

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

def process_documents():
    """Process all documents and create vector store"""
    try:
        processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Process all documents
        documents = processor.process_all_documents(
            material_dir=Config.MATERIAL_DIR,
            questions_dir=Config.QUESTIONS_DIR
        )
        
        if not documents:
            st.error("Nisu pronađeni dokumenti za obradu.")
            return False
        
        # Initialize RAG system
        rag = MLExamRAG(
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            api_version=Config.AZURE_API_VERSION,
            chat_deployment=Config.AZURE_CHAT_DEPLOYMENT,
            embedding_deployment=Config.AZURE_EMBEDDING_DEPLOYMENT
        )
        rag.setup_vectorstore(documents)
        
        st.success(f"Uspešno obrađeno {len(documents)} dokumenata!")
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
        ["📚 Postavi pitanje", "🎯 Generiši kviz", "📝 Vežbaj ispitna pitanja", "⚙️ Podešavanja"]
    )
    
    # Show logout option
    show_logout_option()
    
    # Initialize RAG system
    rag = initialize_rag_system()
    
    if mode == "⚙️ Podešavanja":
        st.header("Podešavanja sistema")
        
        # Document management
        st.subheader("Upravljanje dokumentima")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Materijali za učenje:**")
            material_files = list(Path(Config.MATERIAL_DIR).glob("*.pdf"))
            if material_files:
                for file in material_files:
                    st.write(f"📄 {file.name}")
            else:
                st.write("Nema učitanih materijala")
        
        with col2:
            st.write("**Ispitna pitanja:**")
            question_files = list(Path(Config.QUESTIONS_DIR).glob("*"))
            if question_files:
                for file in question_files:
                    st.write(f"❓ {file.name}")
            else:
                st.write("Nema učitanih pitanja")
        
        st.subheader("Obrada dokumenata")
        st.write("Kliknite da obradite sve dokumente u data/ direktorijumu:")
        
        if st.button("🔄 Obradi dokumente", type="primary"):
            with st.spinner("Obrađujem dokumente..."):
                if process_documents():
                    st.rerun()
        
        # Instructions
        st.subheader("Instrukcije za postavljanje")
        st.markdown("""
        1. **Postavite materijale**: Kopirajte PDF fajlove u `data/material/` direktorijum
        2. **Postavite pitanja**: Kopirajte fajlove sa pitanjima u `data/questions/` direktorijum
        3. **Obradi dokumente**: Kliknite dugme iznad da kreirate vector store
        4. **Azure API**: Uverite se da su Azure OpenAI podaci podešeni u .env fajlu
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

if __name__ == "__main__":
    main()
