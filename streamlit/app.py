import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import ast
import json
import time
import os
import pandas as pd
import logging
from typing import Dict, List, Any
from datetime import datetime
import re

# Configure page
st.set_page_config(
    page_title="🎯 Career Guidance AI - Debug Edition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
def setup_logging():
    """Setup comprehensive logging system"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('RAGCareerAdvisor')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler for detailed logs
    log_file = os.path.join(log_dir, f"rag_debug_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Custom CSS with WCAG AA compliant contrast and modern design
st.markdown("""
<style>
    /* Force light theme for better visibility */
    .stApp {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    
    /* Main header with high contrast fallback */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
        color: #1a1a1a !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Fallback for browsers that don't support background-clip */
    @supports not (-webkit-background-clip: text) {
        .main-header {
            color: #2563eb !important;
            background: none !important;
        }
    }
    
    /* Ensure all text is visible */
    .stMarkdown, .stText, p, div, span {
        color: #1a1a1a !important;
    }
    
    /* Debug console styling */
    .debug-log {
        background-color: #0d1117 !important;
        color: #58a6ff !important;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'SF Mono', 'Monaco', 'Cascadia Code', 'Roboto Mono', monospace;
        font-size: 13px;
        line-height: 1.4;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #30363d;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    }
    
    /* Test result cards */
    .test-result {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .test-pass {
        background-color: #dcfce7 !important;
        border-color: #22c55e !important;
        color: #15803d !important;
    }
    
    .test-fail {
        background-color: #fef2f2 !important;
        border-color: #ef4444 !important;
        color: #dc2626 !important;
    }
    
    .test-warning {
        background-color: #fefce8 !important;
        border-color: #eab308 !important;
        color: #a16207 !important;
    }
    
    /* Metric cards with modern design */
    .metric-card {
        background: #ffffff !important;
        color: #1a1a1a !important;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        background-color: #f8fafc !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Response container */
    .response-container {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        line-height: 1.6;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Primary button override */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        box-shadow: 0 2px 4px rgba(5, 150, 105, 0.3) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.4) !important;
    }
    
    /* Info boxes with better contrast */
    .info-box {
        background-color: #eff6ff !important;
        color: #1e40af !important;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        border: 1px solid #bfdbfe;
    }
    
    .success-box {
        background-color: #f0fdf4 !important;
        color: #166534 !important;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #22c55e;
        margin: 1rem 0;
        border: 1px solid #bbf7d0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc !important;
    }
    
    /* Ensure all form elements are visible */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 2px solid #d1d5db !important;
        border-radius: 8px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Metric styling */
    .css-1xarl3l {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Expandable sections */
    .streamlit-expanderHeader {
        background-color: #f8fafc !important;
        color: #1a1a1a !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #e5e7eb !important;
    }
    
    /* Override any dark theme elements */
    .stApp > div {
        background-color: #ffffff !important;
    }
    
    /* Ensure headers are visible */
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
    }
    
    /* Status indicators with high contrast */
    .status-ready {
        color: #059669 !important;
        font-weight: 600;
    }
    
    .status-warning {
        color: #d97706 !important;
        font-weight: 600;
    }
    
    .status-error {
        color: #dc2626 !important;
        font-weight: 600;
    }
    
    /* Responsive design improvements */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
        
        .response-container {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class DebugRAGEvaluator:
    """Enhanced RAG evaluator with debug capabilities - matches TIM175 evaluation exactly"""
    
    def __init__(self):
        self.evaluator_llm = None
        self.evaluator_embeddings = None
        self.initialized = False
        self.ragas_available = False
        self.evaluation_history = []
        
        # Check if RAGAS is available
        try:
            import ragas
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.ragas_available = True
            logger.info("✅ RAGAS evaluation system available")
        except ImportError as e:
            self.ragas_available = False
            self.import_error = str(e)
            logger.warning(f"⚠️ RAGAS not available: {e}")
    
    def initialize_evaluator(self, google_api_key: str):
        """Initialize RAGAS evaluation components - EXACT config from TIM175 lab"""
        if not self.ragas_available:
            return False, f"❌ RAGAS not available: {self.import_error}"
        
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            # EXACT config from TIM175 evaluation file
            config = {
                "model": "gemini-2.0-flash-lite",
                "temperature": 0.4,
                "max_tokens": None,
                "top_p": 0.8,
            }

            logger.info(f"🔧 Initializing evaluator with config: {config}")

            # Initialize evaluator with Google AI Studio (EXACT same as lab)
            self.evaluator_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                top_p=config["top_p"],
                google_api_key=google_api_key
            ))

            self.evaluator_embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",  # EXACT same model
                task_type="retrieval_document",  # EXACT same task type
                google_api_key=google_api_key
            ))
            
            self.initialized = True
            logger.info("✅ RAG Evaluator initialized successfully")
            return True, "✅ RAG Evaluator initialized!"
            
        except Exception as e:
            logger.error(f"❌ Evaluator initialization failed: {str(e)}")
            return False, f"❌ Evaluator initialization failed: {str(e)}"
    
    def evaluate_response(self, user_query: str, retrieved_contexts: List[str], response: str) -> Dict[str, float]:
        """Evaluate response using EXACT TIM175 evaluation methodology"""
        if not self.ragas_available:
            return {"error": "RAGAS not available", "success": False}
            
        if not self.initialized:
            return {"error": "Evaluator not initialized", "success": False}
        
        try:
            from ragas import evaluate
            from ragas.metrics import Faithfulness, ResponseRelevancy
            from ragas import EvaluationDataset
            
            logger.debug(f"🔍 Evaluating query: {user_query[:100]}...")
            logger.debug(f"📄 Retrieved contexts count: {len(retrieved_contexts)}")
            logger.debug(f"💬 Response length: {len(response)} characters")
            
            # Create evaluation dataset (EXACT format from TIM175 lab)
            dataset = [{
                "user_input": user_query,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
            }]
            
            evaluation_dataset = EvaluationDataset.from_list(dataset)
            
            # Run evaluation (EXACT metrics from TIM175 lab)
            logger.info("📊 Running RAGAS evaluation...")
            result = evaluate(
                dataset=evaluation_dataset, 
                metrics=[Faithfulness(), ResponseRelevancy()], 
                llm=self.evaluator_llm, 
                embeddings=self.evaluator_embeddings
            )
            
            evaluation_result = {
                "faithfulness": float(result['faithfulness'][0]),
                "response_relevancy": float(result['answer_relevancy'][0]),
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in history
            self.evaluation_history.append({
                "query": user_query,
                "result": evaluation_result,
                "context_count": len(retrieved_contexts),
                "response_length": len(response)
            })
            
            logger.info(f"📈 Evaluation complete - Faithfulness: {evaluation_result['faithfulness']:.3f}, Relevancy: {evaluation_result['response_relevancy']:.3f}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {str(e)}")
            return {"error": f"Evaluation failed: {str(e)}", "success": False}


class EnhancedRAGCareerAdvisor:
    """Enhanced RAG system with comprehensive debugging and testing"""
    
    def __init__(self):
        self.embedding_model = None
        self.pinecone_client = None
        self.index = None
        self.initialized = False
        self.debug_logs = []
        self.query_history = []
        
    def log_debug(self, message: str, level: str = "INFO"):
        """Add debug message to logs"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.debug_logs.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.debug_logs) > 100:
            self.debug_logs = self.debug_logs[-100:]
        
        # Also log to file
        if level == "ERROR":
            logger.error(message)
        elif level == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    @st.cache_resource
    def load_embedding_model(_self):
        """Load embedding model with caching - EXACT model from TIM175 lab"""
        logger.info("🤖 Loading embedding model: avsolatorio/GIST-large-Embedding-v0")
        return SentenceTransformer('avsolatorio/GIST-large-Embedding-v0')
        
    def validate_transcript_content(self, content: str, filename: str) -> Dict[str, Any]:
        """Validate transcript content matches expected format"""
        validation_result = {
            "filename": filename,
            "is_valid": True,
            "issues": [],
            "metadata_found": {}
        }
        
        # Check for required fields
        required_patterns = {
            "Interviewee": r'Interviewee:\s*([A-Za-z]+\s+[A-Za-z]+)',
            "Industry Sectors": r'Industry Sectors\s*:\s*([^\n]+?)(?=\s*Takeaways:|#|$)',
            "Source": r'Source\s*:\s*([^\n]+)'
        }
        
        for field, pattern in required_patterns.items():
            match = re.search(pattern, content)
            if match:
                validation_result["metadata_found"][field] = match.group(1).strip()
            else:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Missing {field}")
        
        # Check content length
        if len(content) < 500:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Content too short (< 500 characters)")
        
        # Check for interview-like content
        interview_keywords = ["career", "job", "work", "experience", "advice", "professional"]
        keyword_count = sum(1 for keyword in interview_keywords if keyword.lower() in content.lower())
        
        if keyword_count < 3:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Content doesn't appear to be career interview data")
        
        return validation_result
    
    def rebuild_database_from_local(self):
        """Rebuild database with comprehensive validation and debugging"""
        if not self.initialized:
            return {"error": "Initialize system first", "success": False}
        
        try:
            self.log_debug("🔄 Starting database rebuild from local files")
            
            import os
            from langchain.document_loaders import DirectoryLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.schema import Document
            import re
            
            # EXACT industry sectors from TIM175 lab
            VALID_INDUSTRY_SECTORS = {
                "Not categorized yet", "Architecture and Engineering",
                "Agriculture and Natural Resources", "Marketing, Sales, and Service",
                "Building, Trades, and Construction", "Energy, Environment, Utilities",
                "Fashion and Interior Design", "Manufacturing and Product Development",
                "Education, Child Development, Family Services", "Public and Government Services",
                "Finance and Business", "Arts, Media, and Entertainment",
                "Information and Computer Technologies", "Hospitality, Tourism, Recreation",
                "Health Services, Sciences, Medical Technology"
            }
            
            # Load transcripts from local folder
            transcripts_path = "data/transcripts"
            if not os.path.exists(transcripts_path):
                error_msg = f"Transcripts folder not found at {transcripts_path}"
                self.log_debug(error_msg, "ERROR")
                return {"error": error_msg, "success": False}
            
            self.log_debug(f"📁 Loading transcripts from {transcripts_path}")
            loader = DirectoryLoader(transcripts_path, glob="**/*.txt")
            transcripts = loader.load()
            
            if not transcripts:
                error_msg = "No transcript files found in data/transcripts/"
                self.log_debug(error_msg, "ERROR")
                return {"error": error_msg, "success": False}
            
            self.log_debug(f"✅ Found {len(transcripts)} transcript files")
            st.info(f"📁 Found {len(transcripts)} transcript files")
            
            # Validate transcript content
            validation_results = []
            valid_transcripts = []
            
            for transcript in transcripts:
                filename = os.path.basename(transcript.metadata['source'])
                validation = self.validate_transcript_content(transcript.page_content, filename)
                validation_results.append(validation)
                
                if validation["is_valid"]:
                    valid_transcripts.append(transcript)
                    self.log_debug(f"✅ Valid transcript: {filename}")
                else:
                    self.log_debug(f"⚠️ Invalid transcript: {filename} - Issues: {validation['issues']}", "WARNING")
            
            if not valid_transcripts:
                error_msg = "No valid transcript files found"
                self.log_debug(error_msg, "ERROR")
                return {"error": error_msg, "success": False}
            
            # Text splitting (EXACT parameters from TIM175 lab)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # EXACT same as lab
                chunk_overlap=200  # EXACT same as lab
            )
            
            self.log_debug(f"⚙️ Processing {len(valid_transcripts)} valid transcripts with chunk_size=1000, overlap=200")
            
            # Process chunks (EXACT logic from TIM175 lab)
            chunks = []
            progress_bar = st.progress(0)
            
            for i, transcript in enumerate(valid_transcripts):
                file_path = transcript.metadata['source']
                file_name = os.path.basename(file_path).replace('.txt', '')
                content = transcript.page_content

                # Extract metadata (EXACT regex from TIM175 lab)
                interviewee_match = re.search(r'Interviewee:\s*([A-Za-z]+\s+[A-Za-z]+)', content)
                interviewee_name = interviewee_match.group(1).strip() if interviewee_match else "Unknown"

                sector_matches = re.search(r'Industry Sectors\s*:\s*([^\n]+?)(?=\s*Takeaways:|#|$)', content)
                industry_sector = []
                if sector_matches:
                    extracted_text = sector_matches.group(1)
                    for sector in VALID_INDUSTRY_SECTORS:
                        if sector in extracted_text:
                            industry_sector.append(sector)

                source_match = re.search(r'Source\s*:\s*([^\n]+)', content)
                source = source_match.group(1).strip() if source_match else "Unknown"

                self.log_debug(f"📊 Processing {file_name}: Interviewee={interviewee_name}, Industries={len(industry_sector)}, Source={source[:50]}...")

                # Split text
                splits = text_splitter.split_text(content)
                for j, split in enumerate(splits):
                    chunks.append({
                        "file_name": file_name,
                        "chunk_id": j,
                        "Interviewee": interviewee_name,
                        "Industry Sectors": industry_sector,
                        "Source": source,
                        "content": split,
                    })
                
                progress_bar.progress((i + 1) / len(valid_transcripts))
            
            self.log_debug(f"✅ Created {len(chunks)} chunks from {len(valid_transcripts)} transcripts")
            st.success(f"✅ Processed {len(chunks)} chunks from {len(valid_transcripts)} transcripts")
            
            # Convert to documents
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk["content"],
                    metadata={
                        "file_name": chunk["file_name"],
                        "chunk_id": chunk["chunk_id"],
                        "Interviewee": chunk["Interviewee"],
                        "Industry Sectors": chunk["Industry Sectors"],
                        "Source": chunk["Source"],
                    }
                )
                documents.append(doc)
            
            # Generate embeddings
            self.log_debug("🤖 Generating embeddings...")
            st.info("🤖 Generating embeddings (this may take a few minutes)...")
            embedding_progress = st.progress(0)
            embeddings = []
            
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_embeddings = [self.embedding_model.encode(doc.page_content) for doc in batch]
                embeddings.extend(batch_embeddings)
                embedding_progress.progress(min((i + batch_size) / len(documents), 1.0))
                self.log_debug(f"🔄 Generated embeddings for batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            self.log_debug(f"✅ Generated {len(embeddings)} embeddings")
            st.success(f"✅ Generated {len(embeddings)} embeddings")
            
            # Prepare vectors (EXACT format from TIM175 lab)
            vectors = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                metadata = {
                    "file_name": doc.metadata.get("file_name", f"file_{i}"),
                    "chunk_id": doc.metadata.get("chunk_id", i),
                    "Interviewee": doc.metadata.get("Interviewee", "Unknown"),
                    "Industry Sectors": doc.metadata.get("Industry Sectors", "wrong"),
                    "Source": doc.metadata.get("Source", "Unknown"),
                    "content": doc.page_content,
                }

                vectors.append({
                    "id": f"chunk_{i}",
                    "values": embedding.tolist(),
                    "metadata": metadata
                })
            
            # Upload to Pinecone
            self.log_debug("🚀 Uploading vectors to Pinecone...")
            st.info("🚀 Uploading to Pinecone...")
            upload_progress = st.progress(0)
            
            def upsert_in_batches(vectors, batch_size=25):
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    try:
                        self.index.upsert(vectors=batch)
                        upload_progress.progress((i + batch_size) / len(vectors))
                        self.log_debug(f"✅ Uploaded batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
                    except Exception as e:
                        self.log_debug(f"❌ Error uploading batch {i//batch_size + 1}: {e}", "ERROR")
                        raise
            
            upsert_in_batches(vectors, batch_size=50)
            
            # Wait for indexing
            self.log_debug("⏳ Waiting for Pinecone indexing...")
            st.info("⏳ Waiting for Pinecone to index new data...")
            time.sleep(15)  # Longer wait for better indexing
            
            # Verify upload
            stats = self.index.describe_index_stats()
            self.log_debug(f"📈 Database now contains {stats.total_vector_count} vectors")
            
            return {
                "success": True, 
                "message": f"✅ Successfully rebuilt database!",
                "details": {
                    "transcripts_found": len(transcripts),
                    "valid_transcripts": len(valid_transcripts),
                    "invalid_transcripts": len(transcripts) - len(valid_transcripts),
                    "chunks_created": len(chunks),
                    "vectors_uploaded": len(vectors),
                    "total_database_size": stats.total_vector_count,
                    "validation_results": validation_results
                }
            }
            
        except Exception as e:
            error_msg = f"Rebuild failed: {str(e)}"
            self.log_debug(error_msg, "ERROR")
            return {"error": error_msg, "success": False}
    
    def initialize_models(self, pinecone_api_key: str, google_api_key: str, index_name: str):
        """Initialize models with enhanced debugging"""
        try:
            self.log_debug("🚀 Starting system initialization")
            
            # Initialize embedding model
            with st.spinner("🤖 Loading AI models..."):
                self.embedding_model = self.load_embedding_model()
                self.log_debug("✅ Embedding model loaded successfully")
            
            # Configure Google AI
            genai.configure(api_key=google_api_key)
            self.log_debug("✅ Google AI configured")
            
            # Initialize Pinecone
            with st.spinner("🔗 Connecting to knowledge base..."):
                from pinecone import Pinecone as PineconeClient
                self.pinecone_client = PineconeClient(api_key=pinecone_api_key)
                self.index = self.pinecone_client.Index(index_name)
                
                # Test connection and validate data
                stats = self.index.describe_index_stats()
                self.log_debug(f"📊 Connected to index '{index_name}' with {stats.total_vector_count} vectors")
                
                validation = self.validate_database_content()
                
                if validation.get("issues"):
                    self.log_debug("⚠️ Database validation found issues", "WARNING")
                    st.warning("⚠️ Database validation found potential issues:")
                    for issue in validation["issues"]:
                        st.warning(f"• {issue}")
                    st.info("💡 Consider rebuilding database from local transcript files.")
                else:
                    self.log_debug("✅ Database validation passed")
                
                st.success(f"✅ Connected! Database contains {stats.total_vector_count} career insights")
            
            self.initialized = True
            self.log_debug("🎉 System initialization completed successfully")
            return True, "🎉 System successfully initialized!"
            
        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}"
            self.log_debug(error_msg, "ERROR")
            return False, f"❌ {error_msg}"
    
    def validate_database_content(self) -> Dict[str, Any]:
        """Enhanced database validation with detailed reporting"""
        if not self.initialized:
            return {"error": "Database not initialized"}
        
        try:
            # Query for sample vectors
            stats = self.index.describe_index_stats()
            
            # Test query to sample content
            test_vector = [0.1] * 1024  # Dummy vector for GIST-large-Embedding-v0
            sample_results = self.index.query(
                vector=test_vector,
                top_k=5,
                include_metadata=True
            )
            
            validation_results = {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "sample_count": len(sample_results.get('matches', [])),
                "issues": [],
                "metadata_analysis": {
                    "interviewees_found": 0,
                    "unknown_interviewees": 0,
                    "industries_found": 0,
                    "missing_industries": 0,
                    "avg_content_length": 0,
                    "sources_found": 0
                }
            }
            
            if sample_results.get('matches'):
                content_lengths = []
                interviewees = set()
                industries = set()
                sources = set()
                
                for match in sample_results['matches']:
                    metadata = match['metadata']
                    
                    # Analyze interviewees
                    interviewee = metadata.get('Interviewee', 'Unknown')
                    if interviewee == 'Unknown':
                        validation_results["metadata_analysis"]["unknown_interviewees"] += 1
                    else:
                        validation_results["metadata_analysis"]["interviewees_found"] += 1
                        interviewees.add(interviewee)
                    
                    # Analyze industries
                    industry_data = metadata.get('Industry Sectors', [])
                    if not industry_data or industry_data == "wrong":
                        validation_results["metadata_analysis"]["missing_industries"] += 1
                    else:
                        validation_results["metadata_analysis"]["industries_found"] += 1
                        if isinstance(industry_data, list):
                            industries.update(industry_data)
                        else:
                            industries.add(industry_data)
                    
                    # Analyze content
                    content = metadata.get('content', '')
                    content_lengths.append(len(content))
                    
                    # Analyze sources
                    source = metadata.get('Source', 'Unknown')
                    if source != 'Unknown':
                        validation_results["metadata_analysis"]["sources_found"] += 1
                        sources.add(source)
                
                # Calculate averages
                if content_lengths:
                    validation_results["metadata_analysis"]["avg_content_length"] = sum(content_lengths) / len(content_lengths)
                
                # Flag issues
                if validation_results["metadata_analysis"]["unknown_interviewees"] > 2:
                    validation_results["issues"].append("Many samples have unknown interviewees")
                
                if validation_results["metadata_analysis"]["missing_industries"] > 2:
                    validation_results["issues"].append("Many samples have missing industry classifications")
                
                if validation_results["metadata_analysis"]["avg_content_length"] < 200:
                    validation_results["issues"].append("Content chunks appear too short")
                
                if len(interviewees) < 2:
                    validation_results["issues"].append("Limited variety in interviewees")
                
                # Check for career-related content
                sample_content = " ".join([m['metadata'].get('content', '') for m in sample_results['matches']])
                career_keywords = ['career', 'job', 'work', 'professional', 'interview', 'advice']
                career_keyword_count = sum(1 for keyword in career_keywords if keyword.lower() in sample_content.lower())
                
                if career_keyword_count < 3:
                    validation_results["issues"].append("Content doesn't appear to be career interview data")
                
                # Store sample data
                validation_results["sample_interviewees"] = list(interviewees)[:5]
                validation_results["sample_industries"] = list(industries)[:5]
                validation_results["sample_sources"] = list(sources)[:3]
            else:
                validation_results["issues"].append("No sample data retrieved")
            
            return validation_results
            
        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive tests to validate RAG functionality"""
        self.log_debug("🧪 Starting comprehensive test suite")
        
        test_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": [],
            "overall_status": "UNKNOWN"
        }
        
        # Test queries designed to validate different aspects
        test_queries = [
            {
                "query": "What skills do I need for a career in healthcare?",
                "expected_keywords": ["skills", "healthcare", "medical", "patient", "training"],
                "min_relevance": 0.6,
                "test_name": "Healthcare Skills Query"
            },
            {
                "query": "How do I know if a career is right for me?",
                "expected_keywords": ["career", "right", "fit", "passion", "advice"],
                "min_relevance": 0.5,
                "test_name": "Career Fit Query"
            },
            {
                "query": "What's the daily work like for a teacher?",
                "expected_keywords": ["daily", "work", "teacher", "education", "students"],
                "min_relevance": 0.6,
                "test_name": "Daily Work Query"
            }
        ]
        
        for test_case in test_queries:
            test_results["tests_run"] += 1
            
            try:
                # Run query
                result = self.run_rag_pipeline(test_case["query"], enable_evaluation=False)
                
                if result.get("success"):
                    # Check relevance scores
                    avg_relevance = sum(doc["Score"] for doc in result["retrieved_docs"]) / len(result["retrieved_docs"])
                    
                    # Check for expected keywords in response
                    response_text = result["final_response"].lower()
                    keywords_found = sum(1 for keyword in test_case["expected_keywords"] if keyword in response_text)
                    keyword_score = keywords_found / len(test_case["expected_keywords"])
                    
                    # Determine pass/fail
                    relevance_pass = avg_relevance >= test_case["min_relevance"]
                    keyword_pass = keyword_score >= 0.4  # At least 40% of keywords
                    
                    test_passed = relevance_pass and keyword_pass
                    
                    test_detail = {
                        "test_name": test_case["test_name"],
                        "passed": test_passed,
                        "avg_relevance": avg_relevance,
                        "keyword_score": keyword_score,
                        "keywords_found": keywords_found,
                        "total_keywords": len(test_case["expected_keywords"]),
                        "response_length": len(result["final_response"]),
                        "sources_retrieved": len(result["retrieved_docs"])
                    }
                    
                    if test_passed:
                        test_results["tests_passed"] += 1
                    else:
                        test_results["tests_failed"] += 1
                        test_detail["failure_reasons"] = []
                        if not relevance_pass:
                            test_detail["failure_reasons"].append(f"Low relevance: {avg_relevance:.3f} < {test_case['min_relevance']}")
                        if not keyword_pass:
                            test_detail["failure_reasons"].append(f"Missing keywords: {keyword_score:.1%}")
                    
                else:
                    test_results["tests_failed"] += 1
                    test_detail = {
                        "test_name": test_case["test_name"],
                        "passed": False,
                        "error": result.get("error", "Unknown error")
                    }
                
                test_results["test_details"].append(test_detail)
                
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["test_details"].append({
                    "test_name": test_case["test_name"],
                    "passed": False,
                    "error": str(e)
                })
        
        # Calculate overall status
        pass_rate = test_results["tests_passed"] / test_results["tests_run"] if test_results["tests_run"] > 0 else 0
        
        if pass_rate >= 0.8:
            test_results["overall_status"] = "EXCELLENT"
        elif pass_rate >= 0.6:
            test_results["overall_status"] = "GOOD"
        elif pass_rate >= 0.4:
            test_results["overall_status"] = "NEEDS_IMPROVEMENT"
        else:
            test_results["overall_status"] = "POOR"
        
        self.log_debug(f"🧪 Test suite completed: {test_results['tests_passed']}/{test_results['tests_run']} passed ({pass_rate:.1%})")
        
        return test_results
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a text query"""
        return self.embedding_model.encode(text).tolist()
    
    def api_call(self, prompt: str, model: str = "gemini-2.0-flash-exp") -> str:
        """Make API call to Google's Gemini model"""
        try:
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "max_output_tokens": 2048,
                }
            )
            return response.text
        except Exception as e:
            self.log_debug(f"❌ API call failed: {str(e)}", "ERROR")
            return f"I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def parse_query_string(self, query_string: str) -> Dict[str, Any]:
        """Parse the query string into a dictionary"""
        try:
            # Clean the query string
            query_string = query_string.strip()
            if query_string.startswith('```json'):
                query_string = query_string[7:]
            if query_string.startswith('```'):
                query_string = query_string[3:]
            if query_string.endswith('```'):
                query_string = query_string[:-3]
            
            # Convert the string into a dictionary
            input_dict = ast.literal_eval(query_string)
            
            # Ensure the fields are of the correct type
            result_dict = {
                "content_string_query": input_dict.get("content_string_query", ""),
                "industry_filter": input_dict.get("industry_filter", []),
            }
            
            # Convert industry_filter to lists if they are not already
            if not isinstance(result_dict["industry_filter"], list):
                result_dict["industry_filter"] = [result_dict["industry_filter"]]
            
            return result_dict
        except Exception as e:
            self.log_debug(f"⚠️ Query parsing issue: {e}", "WARNING")
            return {"content_string_query": query_string, "industry_filter": []}
    
    def query_vector_db(self, parsed_dict: Dict[str, Any], top_k: int = 4) -> Dict[str, Any]:
        """Query the vector database for relevant documents"""
        content_string_query = parsed_dict.get("content_string_query", "")
        industry_filter = parsed_dict.get("industry_filter", [])
        
        self.log_debug(f"🔍 Querying database - Query: '{content_string_query[:50]}...', Industries: {industry_filter}")
        
        # Embed user query
        vector = self.embed_query(content_string_query)
        
        # Build query parameters
        query_params = {
            "vector": vector,
            "top_k": top_k,
            "include_values": True,
            "include_metadata": True,
        }
        
        # Add industry filter if specified
        if len(industry_filter) > 0:
            query_params["filter"] = {
                'Industry Sectors': {"$in": industry_filter}
            }
            self.log_debug(f"🏢 Applied industry filter: {industry_filter}")
        
        result = self.index.query(**query_params)
        self.log_debug(f"📊 Retrieved {len(result.get('matches', []))} documents")
        
        return result
    
    def format_documents(self, documents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format retrieved documents for better display"""
        formatted_documents = []
        
        for doc in documents['matches']:
            formatted_doc = {
                "Passage": doc['metadata']['content'],
                "Interviewee": doc['metadata']['Interviewee'],
                "Industry Sectors": doc['metadata']['Industry Sectors'],
                "Source": doc['metadata']['Source'],
                "Score": doc.get('score', 0)
            }
            formatted_documents.append(formatted_doc)
        
        return formatted_documents
    
    def generate_query_object(self, user_query: str) -> str:
        """Convert natural language query to structured query object - EXACT original lab prompt"""
        prompt = f'''
You are a specialized AI assistant that converts high school students' career questions into structured query objects for a vector database search. Your goal is to effectively retrieve relevant career advice from professionals' interviews.

TASK: Transform the student's natural language question into a properly formatted query object.

GUIDELINES FOR OPTIMIZATION:
1. content_string_query:
   - Preserve the FULL semantic meaning of the original question
   - Keep emotional/personal aspects (e.g., "scared", "unprepared", "worried")
   - For multi-part questions, include ALL parts
   - Expand key concepts to improve retrieval (e.g., "scared" → also search for "fear, anxiety, nervous, concerns")
   - Focus on: experiences, challenges, advice, training, skills, daily life, career journey

2. industry_filter (OPTIONAL):
   - Only include if the query explicitly mentions a profession OR clearly implies one
   - Use these exact industry names from our database:
     * Architecture and Engineering
     * Agriculture and Natural Resources
     * Marketing, Sales, and Service
     * Building, Trades, and Construction
     * Energy, Environment, Utilities
     * Fashion and Interior Design
     * Manufacturing and Product Development
     * Education, Child Development, Family Services
     * Public and Government Services
     * Finance and Business
     * Arts, Media, and Entertainment
     * Information and Computer Technologies
     * Hospitality, Tourism, Recreation
     * Health Services, Sciences, Medical Technology

   - Common mappings:
     * firefighter, police → Public and Government Services
     * teacher, professor → Education, Child Development, Family Services
     * nurse, doctor → Health Services, Sciences, Medical Technology

   - If uncertain or query is general career advice, OMIT the industry_filter entirely

FORMATTING RULES:
 - Return ONLY the formatted JSON object with no explanations, no code block formatting (no ```json), no preamble, and no additional text
 - The output should be in this format (without any extra characters, just replace the indicators for curly brackets with the appropriate symbols.):
 (open curly bracket symbol)
   "content_string_query": "your rephrased query here",
   "industry_filter": ["Industry1", "Industry2"]
 (closed curly bracket symbol here)
 - If no industry filter is needed, the format should be:
 (open curly bracket)
   "content_string_query": "your rephrased query here"
 (closed curly bracket)

EXAMPLES:
Input: "What skills did you need to become a firefighter? Were you ever scared?"
Output: {{"content_string_query": "What skills training qualifications needed become firefighter emergency responder scared fear anxiety nervous concerns challenges", "industry_filter": ["Public and Government Services"]}}

Input: "How do I know if a career is right for me?"
Output: {{"content_string_query": "How know career right for me choosing deciding career path advice guidance fit passion interests"}}

Student's query: {user_query}
'''
        
        return self.api_call(prompt)
    
    def generate_response(self, user_query: str, retrieved_context: List[Dict[str, Any]]) -> str:
        """Generate final response using retrieved context - EXACT original lab prompt"""
        prompt = f'''
You are a warm, supportive career guidance counselor helping high school students through real stories from the "What-To-Be" podcast. Your responses should feel like advice from a caring mentor.

TASK: Answer the student's question using ONLY information from the provided professional interviews.

STRICT REQUIREMENTS:
1. FAITHFULNESS: Only use facts explicitly stated in the context. Never add information.
2. RELEVANCE: Directly answer the specific question asked.
3. COMMUNITY: Always introduce professionals by full name, title, and organization.

RESPONSE STRUCTURE:
1. Opening (1-2 sentences):
   - Acknowledge their specific question/concern
   - Show empathy for any emotions expressed

2. Main Response (3-4 paragraphs):
   - DIRECTLY answer their question first
   - Share specific stories/examples from the professionals
   - Include challenges faced and how they overcame them
   - Quote exact words when impactful (use quotation marks)
   - Address each part of multi-part questions

3. Closing (1-2 sentences):
   - Encouragement connected to their concern
   - Practical next step or reflection question

4. Professional Connections:
   Title this section "📎 Meet the Professionals:"
   Format each as:
   - **[Full Name], [Title] at [Organization]**: Listen to their complete journey at [URL]

HANDLING INSUFFICIENT CONTEXT:
If the context doesn't fully answer their question, be honest:
"While the professionals in our interviews don't directly address [specific aspect], they do share insights about [related topic that is covered]..."

TONE GUIDELINES:
- Conversational but respectful
- Encouraging without being dismissive of concerns
- Specific rather than generic
- Focus on storytelling over advice-giving

Remember: These students are making important life decisions. They need authentic stories and honest insights, not generic career advice.

Student Query: {user_query}

Professional Interview Context: {retrieved_context}
'''
        
        return self.api_call(prompt)
    
    def run_rag_pipeline(self, user_query: str, enable_evaluation: bool = False) -> Dict[str, Any]:
        """Complete RAG pipeline with comprehensive logging"""
        if not self.initialized:
            return {"error": "System not initialized. Please check your API keys and settings.", "success": False}
        
        pipeline_start_time = time.time()
        self.log_debug(f"🎯 Starting RAG pipeline for query: '{user_query[:100]}...'")
        
        try:
            # Step 1: Generate query object
            self.log_debug("1️⃣ Generating query object...")
            with st.spinner("🔍 Analyzing your question..."):
                query_string = self.generate_query_object(user_query)
                parsed_dict = self.parse_query_string(query_string)
            
            self.log_debug(f"✅ Query object created: {parsed_dict}")
            
            # Step 2: Retrieve relevant documents
            self.log_debug("2️⃣ Retrieving relevant documents...")
            with st.spinner("📚 Finding relevant career insights..."):
                response = self.query_vector_db(parsed_dict, top_k=4)
                formatted_documents = self.format_documents(response)
            
            avg_relevance = sum(doc["Score"] for doc in formatted_documents) / len(formatted_documents) if formatted_documents else 0
            self.log_debug(f"✅ Retrieved {len(formatted_documents)} documents, avg relevance: {avg_relevance:.3f}")
            
            # Step 3: Generate final response
            self.log_debug("3️⃣ Generating final response...")
            with st.spinner("✨ Creating your personalized advice..."):
                final_response = self.generate_response(user_query, formatted_documents)
            
            self.log_debug(f"✅ Generated response ({len(final_response)} characters)")
            
            result = {
                "query_object": parsed_dict,
                "retrieved_docs": formatted_documents,
                "final_response": final_response,
                "success": True,
                "pipeline_time": time.time() - pipeline_start_time,
                "avg_relevance": avg_relevance
            }
            
            # Step 4: Optional evaluation
            if (enable_evaluation and 
                hasattr(st.session_state, 'rag_evaluator') and 
                st.session_state.rag_evaluator.ragas_available and
                st.session_state.rag_evaluator.initialized):
                
                self.log_debug("4️⃣ Running evaluation...")
                with st.spinner("📊 Evaluating response quality..."):
                    # Prepare contexts for evaluation
                    retrieved_contexts = [doc["Passage"] for doc in formatted_documents]
                    
                    evaluation_result = st.session_state.rag_evaluator.evaluate_response(
                        user_query, retrieved_contexts, final_response
                    )
                    
                    result["evaluation"] = evaluation_result
                    self.log_debug(f"✅ Evaluation complete: {evaluation_result}")
            
            # Store in query history
            self.query_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "result": result
            })
            
            pipeline_time = time.time() - pipeline_start_time
            self.log_debug(f"🎉 Pipeline completed successfully in {pipeline_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.log_debug(error_msg, "ERROR")
            return {"error": error_msg, "success": False}


# Initialize session state
if 'rag_advisor' not in st.session_state:
    st.session_state.rag_advisor = EnhancedRAGCareerAdvisor()
if 'rag_evaluator' not in st.session_state:
    st.session_state.rag_evaluator = DebugRAGEvaluator()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'evaluation_enabled' not in st.session_state:
    st.session_state.evaluation_enabled = False
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

def get_api_keys():
    """Get API keys from environment variables or user input"""
    # Try environment variables first (for deployment)
    pinecone_key = os.getenv('PINECONE_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME', 'pineconedb')
    
    # If not in environment, get from user input
    if not pinecone_key:
        pinecone_key = st.sidebar.text_input("Pinecone API Key", type="password", 
                                           help="Your Pinecone API key")
    else:
        st.sidebar.success("✅ Pinecone API key loaded from environment")
        
    if not google_key:
        google_key = st.sidebar.text_input("Google AI API Key", type="password", 
                                         help="Your Google AI API key")
    else:
        st.sidebar.success("✅ Google AI API key loaded from environment")
    
    if not os.getenv('PINECONE_INDEX_NAME'):
        index_name = st.sidebar.text_input("Pinecone Index Name", value="pineconedb",
                                         help="Name of your Pinecone index")
    else:
        st.sidebar.success(f"✅ Index name: {index_name}")
    
    return pinecone_key, google_key, index_name

def main():
    st.markdown('<h1 class="main-header">🎯 Career Guidance AI - Debug Edition</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
    <strong>Enhanced RAG system with comprehensive debugging, testing, and evaluation</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Get API keys
        pinecone_key, google_key, index_name = get_api_keys()
        
        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox(
            "🐛 Debug Mode", 
            value=st.session_state.debug_mode,
            help="Show detailed debug logs and system information"
        )
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            if pinecone_key and google_key and index_name:
                # Initialize main RAG system
                success, message = st.session_state.rag_advisor.initialize_models(
                    pinecone_key, google_key, index_name
                )
                if success:
                    st.balloons()
                    
                    # Also initialize evaluator if enabled and available
                    if (st.session_state.evaluation_enabled and 
                        st.session_state.rag_evaluator.ragas_available):
                        
                        eval_success, eval_message = st.session_state.rag_evaluator.initialize_evaluator(google_key)
                        if eval_success:
                            st.success("✅ RAG Evaluation system also initialized!")
                        else:
                            st.warning(f"⚠️ Main system ready, but evaluation failed: {eval_message}")
                    elif st.session_state.evaluation_enabled and not st.session_state.rag_evaluator.ragas_available:
                        st.info("💡 Install evaluation packages to enable quality metrics")
                else:
                    st.error(message)
            else:
                st.error("⚠️ Please provide all required credentials.")
        
        # Evaluation settings
        st.markdown("---")
        st.subheader("📊 Evaluation Settings")
        
        # Check if RAGAS is available
        if not st.session_state.rag_evaluator.ragas_available:
            st.warning("""
            ⚠️ **RAG Evaluation Not Available**
            
            Missing required packages. Install with:
            ```bash
            pip install ragas langchain_google_genai
            ```
            """)
            st.session_state.evaluation_enabled = False
        else:
            st.session_state.evaluation_enabled = st.checkbox(
                "🔬 Enable RAG Evaluation (TIM175 Compatible)", 
                value=st.session_state.evaluation_enabled,
                help="Uses exact TIM175 RAGAS metrics: Faithfulness & Response Relevancy"
            )
            
            if st.session_state.evaluation_enabled:
                st.info("""
                **RAGAS Evaluation Metrics (TIM175 Lab):**
                - **Faithfulness**: Response accuracy to context
                - **Response Relevancy**: How well response answers question
                """)
        
        # Testing system
        st.markdown("---")
        st.subheader("🧪 Testing System")
        
        if st.session_state.rag_advisor.initialized:
            if st.button("🧪 Run Test Suite", help="Run comprehensive tests to validate RAG performance"):
                with st.spinner("🧪 Running comprehensive tests..."):
                    test_results = st.session_state.rag_advisor.run_comprehensive_test_suite()
                    st.session_state.test_results = test_results
                st.success(f"✅ Tests completed: {test_results['overall_status']}")
                st.rerun()
        else:
            st.info("Initialize system first to run tests")
        
        # Database management
        st.markdown("---")
        st.subheader("🔄 Database Management")
        
        # Check if local transcript files exist
        transcripts_path = "data/transcripts"
        local_files_exist = os.path.exists(transcripts_path)
        
        if local_files_exist:
            try:
                transcript_files = [f for f in os.listdir(transcripts_path) if f.endswith('.txt')]
                file_count = len(transcript_files)
            except:
                file_count = 0
        else:
            file_count = 0
        
        # Show file status
        if file_count > 0:
            st.success(f"✅ Found {file_count} transcript files")
            
            if st.session_state.rag_advisor.initialized:
                if st.button("🔄 Rebuild Database", type="secondary", help="Rebuild from local transcript files"):
                    with st.spinner("🔄 Rebuilding database..."):
                        result = st.session_state.rag_advisor.rebuild_database_from_local()
                    
                    if result.get("success"):
                        st.success(result["message"])
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(result.get("error", "Unknown error"))
        else:
            st.warning("⚠️ No transcript files found")
            st.info("Add .txt files to `data/transcripts/` folder")
        
        # System status
        st.markdown("---")
        st.subheader("🔧 System Status")
        
        status_items = [
            ("RAG System", "✅ Ready" if st.session_state.rag_advisor.initialized else "⚠️ Not initialized"),
            ("Evaluation", "✅ Ready" if st.session_state.rag_evaluator.initialized else "⚠️ Not ready"),
            ("Debug Mode", "🐛 Enabled" if st.session_state.debug_mode else "📊 Standard mode"),
            ("Transcript Files", f"📁 {file_count} files" if file_count > 0 else "❌ No files")
        ]
        
        for label, status in status_items:
            st.write(f"**{label}**: {status}")
    
    # Main content area
    col1, col2 = st.columns([2, 1] if not st.session_state.debug_mode else [3, 2])
    
    with col1:
        st.subheader("💬 Ask Your Career Question")
        
        # Sample questions
        with st.expander("💡 Test Questions (Designed for Validation)"):
            test_questions = [
                "What skills do I need for a career in healthcare?",
                "How do I know if a career is right for me?",
                "What's the daily work like for a teacher?",
                "I'm worried about choosing the wrong career path. How did professionals know they were making the right choice?",
                "What are ways to explore careers before committing to one path?"
            ]
            
            col_a, col_b = st.columns(2)
            for i, question in enumerate(test_questions):
                with col_a if i % 2 == 0 else col_b:
                    if st.button(f"📝 {question}", key=f"test_q_{i}", use_container_width=True):
                        st.session_state.current_query = question
        
        # Query input
        user_query = st.text_area(
            "Enter your career question:",
            value=st.session_state.get('current_query', ''),
            height=120,
            placeholder="Ask about career paths, skills, daily work, challenges, or any career concerns...",
            help="💡 This system is optimized for career-related questions based on professional interviews"
        )
        
        # Submit button
        submit_col1, submit_col2 = st.columns([3, 1])
        with submit_col1:
            submit_clicked = st.button(
                "🎯 Get Career Advice", 
                type="primary", 
                disabled=not st.session_state.rag_advisor.initialized,
                use_container_width=True
            )
        
        if submit_clicked:
            if user_query.strip():
                # Run RAG pipeline with optional evaluation
                result = st.session_state.rag_advisor.run_rag_pipeline(
                    user_query, 
                    enable_evaluation=st.session_state.evaluation_enabled
                )
                
                # Store in chat history
                st.session_state.chat_history.append({
                    "query": user_query,
                    "result": result,
                    "timestamp": time.time()
                })
                
                # Clear current query
                if 'current_query' in st.session_state:
                    del st.session_state.current_query
                
                st.rerun()
            else:
                st.warning("Please enter a question first!")
    
    with col2:
        # Show test results if available
        if hasattr(st.session_state, 'test_results'):
            st.subheader("🧪 Test Results")
            test_results = st.session_state.test_results
            
            # Overall status
            status_color = {
                "EXCELLENT": "🟢",
                "GOOD": "🟡", 
                "NEEDS_IMPROVEMENT": "🟠",
                "POOR": "🔴"
            }
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{status_color.get(test_results['overall_status'], '❓')} {test_results['overall_status']}</h4>
                <p>Passed: {test_results['tests_passed']}/{test_results['tests_run']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Individual test results
            with st.expander("📋 Detailed Test Results"):
                for test in test_results['test_details']:
                    if test.get('passed'):
                        st.markdown(f'<div class="test-result test-pass">✅ <strong>{test["test_name"]}</strong><br>Relevance: {test.get("avg_relevance", 0):.3f}<br>Keywords: {test.get("keyword_score", 0):.1%}</div>', unsafe_allow_html=True)
                    else:
                        failure_reasons = test.get('failure_reasons', [test.get('error', 'Unknown error')])
                        st.markdown(f'<div class="test-result test-fail">❌ <strong>{test["test_name"]}</strong><br>{"<br>".join(failure_reasons)}</div>', unsafe_allow_html=True)
        
        # Latest query analysis
        if st.session_state.chat_history:
            st.subheader("📈 Latest Query Analysis")
            last_result = st.session_state.chat_history[-1]["result"]
            
            if last_result.get("success"):
                # Show metrics
                st.metric("⏱️ Pipeline Time", f"{last_result.get('pipeline_time', 0):.2f}s")
                st.metric("🎯 Avg Relevance", f"{last_result.get('avg_relevance', 0):.3f}")
                st.metric("📄 Sources Found", len(last_result.get("retrieved_docs", [])))
                
                # Show evaluation if available
                if "evaluation" in last_result and last_result["evaluation"].get("success"):
                    eval_data = last_result["evaluation"]
                    st.write("**📊 RAGAS Scores:**")
                    st.write(f"• Faithfulness: {eval_data.get('faithfulness', 0):.3f}")
                    st.write(f"• Relevancy: {eval_data.get('response_relevancy', 0):.3f}")
        
        # Debug logs (if debug mode enabled)
        if st.session_state.debug_mode and st.session_state.rag_advisor.debug_logs:
            st.subheader("🐛 Debug Logs")
            
            # Show recent debug logs
            recent_logs = st.session_state.rag_advisor.debug_logs[-20:]  # Last 20 entries
            log_text = "\n".join(recent_logs)
            
            st.markdown(f'<div class="debug-log">{log_text}</div>', unsafe_allow_html=True)
            
            if st.button("📥 Download Full Logs"):
                log_content = "\n".join(st.session_state.rag_advisor.debug_logs)
                st.download_button(
                    label="💾 Download Debug Logs",
                    data=log_content,
                    file_name=f"rag_debug_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    # Display conversation history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💭 Recent Conversations")
        
        # Show most recent conversations
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
            with st.container():
                # User question
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <strong>🧑‍🎓 You asked:</strong><br>
                    {chat["query"]}
                </div>
                """, unsafe_allow_html=True)
                
                # AI response
                result = chat["result"]
                if result.get("success"):
                    st.markdown(f"""
                    <div style="background-color: white; padding: 1.5rem; border-radius: 10px; border: 1px solid #ddd; margin: 1rem 0;">
                        <strong>🤖 Career Advisor:</strong><br><br>
                        {result["final_response"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show evaluation metrics if available
                    if "evaluation" in result and result["evaluation"].get("success"):
                        eval_data = result["evaluation"]
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            faithfulness = eval_data.get("faithfulness", 0)
                            color = "🟢" if faithfulness > 0.8 else "🟡" if faithfulness > 0.6 else "🔴"
                            st.metric("📖 Faithfulness", f"{color} {faithfulness:.2f}")
                        
                        with col2:
                            relevancy = eval_data.get("response_relevancy", 0)
                            color = "🟢" if relevancy > 0.8 else "🟡" if relevancy > 0.6 else "🔴"
                            st.metric("🎯 Relevancy", f"{color} {relevancy:.2f}")
                        
                        with col3:
                            overall = (faithfulness + relevancy) / 2
                            color = "🟢" if overall > 0.8 else "🟡" if overall > 0.6 else "🔴"
                            st.metric("⭐ Overall", f"{color} {overall:.2f}")
                    
                    # Debug info expandable
                    with st.expander(f"🔍 Technical Details (Pipeline: {result.get('pipeline_time', 0):.2f}s)"):
                        # Query processing
                        query_obj = result.get("query_object", {})
                        st.write("**Query Processing:**")
                        st.write(f"• Processed: `{query_obj.get('content_string_query', 'N/A')}`")
                        st.write(f"• Industry Filter: {query_obj.get('industry_filter', 'None')}")
                        
                        # Retrieved sources
                        st.write("**Retrieved Sources:**")
                        for j, doc in enumerate(result.get("retrieved_docs", []), 1):
                            score_color = "🟢" if doc['Score'] > 0.8 else "🟡" if doc['Score'] > 0.6 else "🔴"
                            st.write(f"{score_color} **{j}. {doc['Interviewee']}** ({doc['Industry Sectors']}) - Score: {doc['Score']:.3f}")
                
                else:
                    st.error(f"❌ {result.get('error', 'Unknown error')}")
                
                st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    