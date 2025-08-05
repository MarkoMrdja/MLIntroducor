"""
Azure Blob Storage Document Loader
Handles loading and managing documents from Azure Blob Storage with incremental updates
"""

import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import tempfile

from azure.storage.blob import BlobServiceClient, BlobClient
from langchain_core.documents import Document

from config import Config


class AzureBlobDocumentLoader:
    def __init__(self, connection_string: str, container_name: str):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
        self.container_client = self.blob_service_client.get_container_client(container_name)
        
        # Local metadata tracking
        self.metadata_file = "./blob_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load metadata about previously processed documents"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.processed_docs = json.load(f)
        else:
            self.processed_docs = {}
    
    def save_metadata(self):
        """Save metadata about processed documents"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_docs, f, indent=2, ensure_ascii=False)
    
    def get_blob_hash(self, blob_client: BlobClient) -> str:
        """Get hash of blob content for change detection"""
        properties = blob_client.get_blob_properties()
        # Use last modified time and size as a simple hash
        content_info = f"{properties.last_modified}_{properties.size}"
        return hashlib.md5(content_info.encode()).hexdigest()
    
    def list_blobs_by_prefix(self, prefix: str) -> List[Dict]:
        """List all blobs with given prefix"""
        blobs = []
        blob_list = self.container_client.list_blobs(name_starts_with=prefix)
        
        for blob in blob_list:
            # Skip directories and only include PDF files
            if (blob.name.lower().endswith('.pdf') and 
                not blob.name.endswith('/') and 
                blob.size > 0):  # Ensure it's actually a file, not a directory
                
                try:
                    blob_client = self.container_client.get_blob_client(blob.name)
                    blob_hash = self.get_blob_hash(blob_client)
                    
                    blobs.append({
                        'name': blob.name,
                        'size': blob.size,
                        'last_modified': blob.last_modified.isoformat(),
                        'hash': blob_hash,
                        'full_path': blob.name
                    })
                except Exception as e:
                    print(f"Error processing blob {blob.name}: {str(e)}")
                    continue
        
        return blobs
    
    def get_new_or_updated_documents(self) -> Tuple[List[Dict], List[str]]:
        """
        Get list of documents that are new or have been updated
        Returns: (new_or_updated_blobs, removed_document_ids)
        """
        # Get current blobs
        material_blobs = self.list_blobs_by_prefix(Config.BLOB_MATERIAL_PREFIX)
        question_blobs = self.list_blobs_by_prefix(Config.BLOB_QUESTIONS_PREFIX)
        current_blobs = material_blobs + question_blobs
        
        new_or_updated = []
        current_blob_names = set()
        
        for blob in current_blobs:
            current_blob_names.add(blob['name'])
            blob_id = self.get_document_id(blob['name'])
            
            # Check if this is a new document or if it has been updated
            if (blob_id not in self.processed_docs or 
                self.processed_docs[blob_id]['hash'] != blob['hash']):
                new_or_updated.append(blob)
        
        # Find documents that were removed from blob storage
        removed_ids = []
        for doc_id, doc_info in self.processed_docs.items():
            if doc_info['blob_name'] not in current_blob_names:
                removed_ids.append(doc_id)
        
        return new_or_updated, removed_ids
    
    def get_document_id(self, blob_name: str) -> str:
        """Generate a consistent document ID from blob name"""
        return hashlib.md5(blob_name.encode()).hexdigest()
    
    def download_and_process_blob(self, blob_info: Dict) -> Optional[str]:
        """Download blob to temporary file and return the path"""
        try:
            blob_client = self.container_client.get_blob_client(blob_info['name'])
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                blob_data = blob_client.download_blob()
                blob_data.readinto(temp_file)
                temp_file_path = temp_file.name
            
            return temp_file_path
        
        except Exception as e:
            print(f"Error downloading blob {blob_info['name']}: {str(e)}")
            return None
    
    def update_processed_metadata(self, blob_info: Dict, chunk_count: int):
        """Update metadata for a processed document"""
        doc_id = self.get_document_id(blob_info['name'])
        self.processed_docs[doc_id] = {
            'blob_name': blob_info['name'],
            'hash': blob_info['hash'],
            'processed_at': datetime.now().isoformat(),
            'chunk_count': chunk_count,
            'size': blob_info['size']
        }
        self.save_metadata()
    
    def remove_processed_metadata(self, doc_ids: List[str]):
        """Remove metadata for documents that no longer exist"""
        for doc_id in doc_ids:
            if doc_id in self.processed_docs:
                del self.processed_docs[doc_id]
        if doc_ids:
            self.save_metadata()
    
    def get_all_current_documents(self) -> List[Dict]:
        """Get all current documents from blob storage (for full rebuild)"""
        material_blobs = self.list_blobs_by_prefix(Config.BLOB_MATERIAL_PREFIX)
        question_blobs = self.list_blobs_by_prefix(Config.BLOB_QUESTIONS_PREFIX)
        return material_blobs + question_blobs
    
    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary file"""
        try:
            os.unlink(file_path)
        except:
            pass
    
    def get_processing_stats(self) -> Dict:
        """Get statistics about processed documents and available documents"""
        # Processed documents stats
        total_processed = len(self.processed_docs)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in self.processed_docs.values())
        
        material_processed = sum(1 for doc in self.processed_docs.values() 
                          if doc['blob_name'].startswith(Config.BLOB_MATERIAL_PREFIX))
        question_processed = sum(1 for doc in self.processed_docs.values() 
                          if doc['blob_name'].startswith(Config.BLOB_QUESTIONS_PREFIX))
        
        # Available documents stats (from blob storage)
        try:
            material_blobs = self.list_blobs_by_prefix(Config.BLOB_MATERIAL_PREFIX)
            question_blobs = self.list_blobs_by_prefix(Config.BLOB_QUESTIONS_PREFIX)
            
            total_available = len(material_blobs) + len(question_blobs)
            material_available = len(material_blobs)
            question_available = len(question_blobs)
        except Exception as e:
            print(f"Error getting available documents: {str(e)}")
            total_available = material_available = question_available = 0
        
        return {
            'total_documents': total_processed,
            'total_chunks': total_chunks,
            'material_documents': material_processed,
            'question_documents': question_processed,
            'last_update': max([doc.get('processed_at', '') for doc in self.processed_docs.values()], default='Never'),
            # Add available documents info
            'total_available': total_available,
            'material_available': material_available,
            'question_available': question_available
        }


class IncrementalDocumentProcessor:
    """Handles incremental document processing with Azure Blob Storage"""
    
    def __init__(self, document_processor, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.document_processor = document_processor
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize blob loader
        self.blob_loader = AzureBlobDocumentLoader(
            Config.AZURE_STORAGE_CONNECTION_STRING,
            Config.AZURE_STORAGE_CONTAINER_NAME
        )
    
    def process_incremental_update(self) -> Tuple[List[Document], Dict]:
        """
        Process only new or updated documents
        Returns: (new_documents, processing_stats)
        """
        new_or_updated_blobs, removed_doc_ids = self.blob_loader.get_new_or_updated_documents()
        
        stats = {
            'new_or_updated': len(new_or_updated_blobs),
            'removed': len(removed_doc_ids),
            'processed_successfully': 0,
            'processing_errors': [],
            'removed_doc_ids': removed_doc_ids
        }
        
        new_documents = []
        
        # Process new or updated documents
        for blob_info in new_or_updated_blobs:
            try:
                # Download blob to temporary file
                temp_file_path = self.blob_loader.download_and_process_blob(blob_info)
                if not temp_file_path:
                    continue
                
                # Determine document type and process accordingly
                if blob_info['name'].startswith(Config.BLOB_MATERIAL_PREFIX):
                    doc_type = 'material'
                elif blob_info['name'].startswith(Config.BLOB_QUESTIONS_PREFIX):
                    doc_type = 'questions'
                else:
                    doc_type = 'unknown'
                
                # Process the document
                docs = self.document_processor.process_single_document(
                    temp_file_path, 
                    doc_type=doc_type,
                    source_name=blob_info['name']
                )
                
                new_documents.extend(docs)
                
                # Update metadata
                self.blob_loader.update_processed_metadata(blob_info, len(docs))
                stats['processed_successfully'] += 1
                
                # Clean up temp file
                self.blob_loader.cleanup_temp_file(temp_file_path)
                
            except Exception as e:
                error_msg = f"Error processing {blob_info['name']}: {str(e)}"
                stats['processing_errors'].append(error_msg)
                print(error_msg)
        
        # Remove metadata for deleted documents
        if removed_doc_ids:
            self.blob_loader.remove_processed_metadata(removed_doc_ids)
        
        return new_documents, stats
    
    def process_full_rebuild(self) -> Tuple[List[Document], Dict]:
        """
        Process all documents (full rebuild)
        Returns: (all_documents, processing_stats)
        """
        all_blobs = self.blob_loader.get_all_current_documents()
        
        stats = {
            'total_documents': len(all_blobs),
            'processed_successfully': 0,
            'processing_errors': []
        }
        
        all_documents = []
        
        # Clear existing metadata
        self.blob_loader.processed_docs = {}
        
        for blob_info in all_blobs:
            try:
                # Download and process
                temp_file_path = self.blob_loader.download_and_process_blob(blob_info)
                if not temp_file_path:
                    continue
                
                # Determine document type
                if blob_info['name'].startswith(Config.BLOB_MATERIAL_PREFIX):
                    doc_type = 'material'
                elif blob_info['name'].startswith(Config.BLOB_QUESTIONS_PREFIX):
                    doc_type = 'questions'
                else:
                    doc_type = 'unknown'
                
                # Process the document
                docs = self.document_processor.process_single_document(
                    temp_file_path,
                    doc_type=doc_type,
                    source_name=blob_info['name']
                )
                
                all_documents.extend(docs)
                
                # Update metadata
                self.blob_loader.update_processed_metadata(blob_info, len(docs))
                stats['processed_successfully'] += 1
                
                # Clean up
                self.blob_loader.cleanup_temp_file(temp_file_path)
                
            except Exception as e:
                error_msg = f"Error processing {blob_info['name']}: {str(e)}"
                stats['processing_errors'].append(error_msg)
                print(error_msg)
        
        return all_documents, stats
    
    def get_storage_stats(self) -> Dict:
        """Get statistics about blob storage and processed documents"""
        return self.blob_loader.get_processing_stats()
