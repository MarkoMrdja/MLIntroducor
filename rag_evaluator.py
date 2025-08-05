"""
RAG System Evaluator
Provides tools to evaluate and monitor RAG system performance
"""

import json
import time
from typing import List, Dict, Any
from pathlib import Path
import streamlit as st
from rag_system import MLExamRAG

class RAGEvaluator:
    def __init__(self, rag_system: MLExamRAG):
        self.rag = rag_system
        self.evaluation_log = []
    
    def evaluate_retrieval_quality(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Evaluate how well the retrieval system works for a given query"""
        start_time = time.time()
        
        # Get retrieval results with scores
        try:
            # This method needs to be implemented in rag_system.py
            results = self.rag.vectorstore.similarity_search_with_score(query, k=k)
            retrieval_time = time.time() - start_time
            
            evaluation = {
                "query": query,
                "retrieval_time": retrieval_time,
                "num_results": len(results),
                "results": []
            }
            
            for i, (doc, score) in enumerate(results):
                result_info = {
                    "rank": i + 1,
                    "similarity_score": float(score),
                    "content_preview": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "content_length": len(doc.page_content)
                }
                evaluation["results"].append(result_info)
            
            return evaluation
            
        except Exception as e:
            return {"error": str(e), "query": query}
    
    def evaluate_answer_quality(self, question: str, answer: str) -> Dict[str, Any]:
        """Evaluate the quality of an answer using the RAG system itself"""
        evaluation_prompt = f"""
        Oceni kvalitet ovog odgovora na pitanje iz mašinskog učenja:
        
        PITANJE: {question}
        
        ODGOVOR: {answer}
        
        Oceni sledeće (skala 1-5):
        1. Tačnost odgovora
        2. Kompletnost odgovora  
        3. Jasnoća objašnjenja
        4. Korišćenje relevantnih termina
        
        Format odgovora:
        Tačnost: X/5
        Kompletnost: X/5
        Jasnoća: X/5
        Terminologija: X/5
        
        Kratko objašnjenje ocene:
        """
        
        try:
            evaluation = self.rag.llm.invoke(evaluation_prompt).content
            return {
                "question": question,
                "answer": answer,
                "evaluation": evaluation,
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def run_test_queries(self) -> List[Dict[str, Any]]:
        """Run a set of test queries to evaluate system performance"""
        test_queries = [
            "Šta je razlika između supervised i unsupervised learning?",
            "Objasni kako funkcioniše SVM algoritam",
            "Šta je cross-validation i zašto se koristi?",
            "Objasni bias-variance tradeoff",
            "Kako funkcioniše k-means klasterizacija?",
            "Šta je PCA i kada se koristi?",
            "Objasni kako funkcioniše neural network",
            "Šta je regularizacija u mašinskom učenju?"
        ]
        
        results = []
        for query in test_queries:
            # Test retrieval
            retrieval_eval = self.evaluate_retrieval_quality(query)
            
            # Test full answer generation
            try:
                start_time = time.time()
                answer = self.rag.ask_question(query, mode="explanation")
                generation_time = time.time() - start_time
                
                # Evaluate answer quality
                answer_eval = self.evaluate_answer_quality(query, answer)
                
                result = {
                    "query": query,
                    "retrieval_evaluation": retrieval_eval,
                    "answer": answer,
                    "generation_time": generation_time,
                    "answer_evaluation": answer_eval
                }
                
            except Exception as e:
                result = {
                    "query": query,
                    "error": str(e),
                    "retrieval_evaluation": retrieval_eval
                }
            
            results.append(result)
            
        return results
    
    def display_retrieval_analysis(self, query: str):
        """Display retrieval analysis in Streamlit"""
        st.subheader("🔍 Analiza pronalaska dokumenata")
        
        eval_result = self.evaluate_retrieval_quality(query, k=5)
        
        if "error" in eval_result:
            st.error(f"Greška: {eval_result['error']}")
            return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vreme pretrage", f"{eval_result['retrieval_time']:.3f}s")
        with col2:
            st.metric("Broj rezultata", eval_result['num_results'])
        with col3:
            avg_score = sum(r['similarity_score'] for r in eval_result['results']) / len(eval_result['results'])
            st.metric("Prosečna sličnost", f"{avg_score:.3f}")
        
        st.subheader("Pronađeni dokumenti:")
        for result in eval_result['results']:
            with st.expander(f"#{result['rank']} - Sličnost: {result['similarity_score']:.3f}"):
                st.write("**Izvor:**", result['metadata'].get('source', 'Nepoznat'))
                st.write("**Sadržaj:**")
                st.text(result['content_preview'])
                st.write("**Dužina:**", result['content_length'], "karaktera")
    
    def display_system_stats(self):
        """Display overall system statistics"""
        try:
            # Get collection info
            collection = self.rag.vectorstore._collection
            doc_count = collection.count()
            
            st.subheader("📊 Statistike sistema")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ukupno dokumenata u bazi", doc_count)
            with col2:
                st.metric("Veličina chunk-a", self.rag.chunk_size if hasattr(self.rag, 'chunk_size') else "N/A")
            
            # Show some sample documents
            if doc_count > 0:
                st.subheader("Uzorci dokumenata u bazi:")
                sample_results = self.rag.vectorstore.similarity_search("machine learning", k=3)
                for i, doc in enumerate(sample_results[:3]):
                    with st.expander(f"Uzorak #{i+1}"):
                        st.write("**Izvor:**", doc.metadata.get('source', 'Nepoznat'))
                        st.text(doc.page_content[:300] + "...")
                        
        except Exception as e:
            st.error(f"Greška pri učitavanju statistika: {str(e)}")

def create_evaluation_dashboard():
    """Create Streamlit dashboard for RAG evaluation"""
    st.header("🧪 Evaluacija RAG Sistema")
    
    st.markdown("""
    Ova sekcija vam pomaže da ocenite koliko dobro radi vaš RAG sistem.
    Možete testirati kako sistem pronalazi relevantne dokumente i kako generiše odgovore.
    """)
    
    # Initialize evaluator
    try:
        from app import initialize_rag_system
        rag = initialize_rag_system()
        
        if rag is None:
            st.error("RAG sistem nije inicijalizovan.")
            return
        
        evaluator = RAGEvaluator(rag)
        
        tab1, tab2, tab3 = st.tabs(["🔍 Test pretrage", "📊 Statistike", "🧪 Batch testiranje"])
        
        with tab1:
            st.subheader("Testirajte pretragu dokumenata")
            test_query = st.text_input(
                "Unesite pitanje za testiranje:",
                value="Šta je SVM algoritam?",
                help="Sistem će prikazati koje dokumente pronađe za ovo pitanje"
            )
            
            if st.button("Analiziraj pretragu"):
                evaluator.display_retrieval_analysis(test_query)
        
        with tab2:
            evaluator.display_system_stats()
        
        with tab3:
            st.subheader("Testiraj sistem sa unapred definisanim pitanjima")
            if st.button("Pokreni batch test"):
                with st.spinner("Testiram sistem..."):
                    results = evaluator.run_test_queries()
                    
                    successful_tests = [r for r in results if "error" not in r]
                    failed_tests = [r for r in results if "error" in r]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"Uspešno: {len(successful_tests)}/{len(results)}")
                    with col2:
                        if failed_tests:
                            st.error(f"Neuspešno: {len(failed_tests)}")
                    
                    # Show results
                    for result in successful_tests[:3]:  # Show first 3
                        with st.expander(f"Pitanje: {result['query'][:50]}..."):
                            st.write("**Vreme generisanja:**", f"{result['generation_time']:.2f}s")
                            st.write("**Odgovor:**")
                            st.text(result['answer'][:500] + "...")
                            
                            if "answer_evaluation" in result and "error" not in result["answer_evaluation"]:
                                st.write("**Automatska ocena:**")
                                st.text(result["answer_evaluation"]["evaluation"][:300] + "...")
                    
                    if failed_tests:
                        st.subheader("❌ Neuspešni testovi:")
                        for result in failed_tests:
                            st.error(f"Pitanje: {result['query']} | Greška: {result['error']}")
        
    except Exception as e:
        st.error(f"Greška pri pokretanju evaluatora: {str(e)}")
