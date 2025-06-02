import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import ast
import json
import time
import os
from typing import Dict, List, Any

# Configure page
st.set_page_config(
    page_title="🎯 Career Guidance AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    .response-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class RAGCareerAdvisor:
    def __init__(self):
        self.embedding_model = None
        self.pinecone_client = None
        self.index = None
        self.initialized = False
        
    @st.cache_resource
    def load_embedding_model(_self):
        """Load embedding model with caching"""
        return SentenceTransformer('avsolatorio/GIST-large-Embedding-v0')
        
    def initialize_models(self, pinecone_api_key: str, google_api_key: str, index_name: str):
        """Initialize the embedding model and Pinecone connection"""
        try:
            # Initialize embedding model
            with st.spinner("🤖 Loading AI models..."):
                self.embedding_model = self.load_embedding_model()
            
            # Configure Google AI
            genai.configure(api_key=google_api_key)
            
            # Initialize Pinecone
            with st.spinner("🔗 Connecting to knowledge base..."):
                self.pinecone_client = Pinecone(api_key=pinecone_api_key)
                self.index = self.pinecone_client.Index(index_name)
                
                # Test connection
                stats = self.index.describe_index_stats()
                st.success(f"✅ Connected! Database contains {stats.total_vector_count} career insights")
            
            self.initialized = True
            return True, "🎉 System successfully initialized!"
            
        except Exception as e:
            return False, f"❌ Initialization failed: {str(e)}"
    
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
            st.error(f"API Error: {str(e)}")
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
            st.warning(f"Query parsing issue: {e}")
            return {"content_string_query": query_string, "industry_filter": []}
    
    def query_vector_db(self, parsed_dict: Dict[str, Any], top_k: int = 4) -> Dict[str, Any]:
        """Query the vector database for relevant documents"""
        content_string_query = parsed_dict.get("content_string_query", "")
        industry_filter = parsed_dict.get("industry_filter", [])
        
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
        
        return self.index.query(**query_params)
    
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
        """Convert natural language query to structured query object"""
        prompt = f'''
You are a specialized AI assistant that converts high school students' career questions into structured query objects for a vector database search.

TASK: Transform the student's question into a properly formatted query object.

GUIDELINES:
1. content_string_query: Preserve the full meaning, include emotional aspects, expand key concepts
2. industry_filter (OPTIONAL): Only include if the query explicitly mentions a profession

Available industries:
- Architecture and Engineering
- Agriculture and Natural Resources  
- Marketing, Sales, and Service
- Building, Trades, and Construction
- Energy, Environment, Utilities
- Fashion and Interior Design
- Manufacturing and Product Development
- Education, Child Development, Family Services
- Public and Government Services
- Finance and Business
- Arts, Media, and Entertainment
- Information and Computer Technologies
- Hospitality, Tourism, Recreation
- Health Services, Sciences, Medical Technology

FORMATTING: Return ONLY a valid Python dictionary (no markdown, no explanations).

EXAMPLES:
Input: "What skills do firefighters need? Were you scared?"
Output: {{"content_string_query": "skills training qualifications firefighter emergency responder scared fear anxiety challenges", "industry_filter": ["Public and Government Services"]}}

Input: "How do I choose the right career?"
Output: {{"content_string_query": "choose right career path decision making advice guidance"}}

Student query: {user_query}
'''
        
        return self.api_call(prompt)
    
    def generate_response(self, user_query: str, retrieved_context: List[Dict[str, Any]]) -> str:
        """Generate final response using retrieved context"""
        prompt = f'''
You are a warm, supportive career counselor helping high school students using real stories from professional interviews.

TASK: Answer the student's question using ONLY the provided interview context.

REQUIREMENTS:
1. Be faithful to the context - only use stated facts
2. Directly address their specific question
3. Include professional names and organizations
4. Use conversational, encouraging tone

STRUCTURE:
1. Acknowledge their question with empathy
2. Answer directly using specific examples from the interviews
3. Include relevant quotes when impactful
4. End with encouragement and next steps
5. List the professionals: "📎 Meet the Professionals:" with names, titles, and sources

Student Question: {user_query}

Professional Interview Context: {retrieved_context}
'''
        
        return self.api_call(prompt)
    
    def run_rag_pipeline(self, user_query: str) -> Dict[str, Any]:
        """Complete RAG pipeline"""
        if not self.initialized:
            return {"error": "System not initialized. Please check your API keys and settings.", "success": False}
        
        try:
            # Step 1: Generate query object
            with st.spinner("🔍 Analyzing your question..."):
                query_string = self.generate_query_object(user_query)
                parsed_dict = self.parse_query_string(query_string)
            
            # Step 2: Retrieve relevant documents
            with st.spinner("📚 Finding relevant career insights..."):
                response = self.query_vector_db(parsed_dict, top_k=4)
                formatted_documents = self.format_documents(response)
            
            # Step 3: Generate final response
            with st.spinner("✨ Creating your personalized advice..."):
                final_response = self.generate_response(user_query, formatted_documents)
            
            return {
                "query_object": parsed_dict,
                "retrieved_docs": formatted_documents,
                "final_response": final_response,
                "success": True
            }
            
        except Exception as e:
            return {"error": f"Pipeline error: {str(e)}", "success": False}

# Initialize session state
if 'rag_advisor' not in st.session_state:
    st.session_state.rag_advisor = RAGCareerAdvisor()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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
    st.markdown('<h1 class="main-header">🎯 Career Guidance AI</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>💡 Welcome!</strong> Ask questions about career paths, required skills, work-life balance, 
    education requirements, or any career concerns. Get advice from real professionals who've been there!
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Setup")
        
        # Get API keys
        pinecone_key, google_key, index_name = get_api_keys()
        
        # Initialize button
        if st.button("🚀 Initialize System", type="primary"):
            if pinecone_key and google_key and index_name:
                success, message = st.session_state.rag_advisor.initialize_models(
                    pinecone_key, google_key, index_name
                )
                if success:
                    st.balloons()
                else:
                    st.error(message)
            else:
                st.error("⚠️ Please provide all required credentials.")
        
        # System status
        st.markdown("---")
        st.subheader("🔧 System Status")
        if st.session_state.rag_advisor.initialized:
            st.markdown('<div class="success-box">✅ <strong>Ready to help!</strong></div>', 
                       unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please initialize the system first")
        
        # Statistics
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("📊 Session Stats")
            st.metric("Questions Asked", len(st.session_state.chat_history))
        
        # Clear chat button
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # About
        st.markdown("---")
        with st.expander("ℹ️ About This App"):
            st.markdown("""
            This app uses AI to provide career guidance based on real interviews 
            with professionals from the "What-To-Be" podcast. 
            
            **How it works:**
            1. You ask a career question
            2. AI finds relevant professional stories
            3. You get personalized advice based on real experiences
            
            **Data Source:** Professional interviews covering 14+ industry sectors
            """)
    
    # Main interface
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("💬 Ask Your Career Question")
        
        # Sample questions
        with st.expander("💡 Try These Sample Questions"):
            sample_questions = [
                "I'm worried about choosing the wrong career. How did professionals know they were making the right choice?",
                "What skills do I need to become a firefighter? Were you ever scared during training?",
                "How do professionals balance work life with personal time?",
                "Is college really necessary for career success?",
                "What are ways to explore careers before committing to one path?"
            ]
            
            for i, question in enumerate(sample_questions, 1):
                if st.button(f"📝 {question}", key=f"sample_{i}", use_container_width=True):
                    st.session_state.current_query = question
        
        # Query input
        user_query = st.text_area(
            "Type your question here:",
            value=st.session_state.get('current_query', ''),
            height=120,
            placeholder="e.g., 'I'm interested in healthcare but scared of blood. How do medical professionals handle their fears?'",
            help="Ask about anything career-related - skills, education, challenges, day-to-day work, etc."
        )
        
        # Submit button
        submit_col1, submit_col2 = st.columns([3, 1])
        with submit_col1:
            submit_clicked = st.button("🎯 Get Career Advice", type="primary", 
                                     disabled=not st.session_state.rag_advisor.initialized,
                                     use_container_width=True)
        
        if submit_clicked:
            if user_query.strip():
                # Run RAG pipeline
                with st.spinner("🤔 Thinking..."):
                    result = st.session_state.rag_advisor.run_rag_pipeline(user_query)
                
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
        if st.session_state.chat_history:
            st.subheader("📈 Latest Query Insights")
            last_result = st.session_state.chat_history[-1]["result"]
            
            if last_result.get("success") and "retrieved_docs" in last_result:
                # Show relevant sources
                st.metric("🔍 Sources Found", len(last_result["retrieved_docs"]))
                
                # Show top industries
                industries = set()
                for doc in last_result["retrieved_docs"]:
                    sectors = doc["Industry Sectors"]
                    if isinstance(sectors, list):
                        industries.update(sectors)
                    else:
                        industries.add(sectors)
                
                industries = [i for i in industries if i != "wrong" and i != "Not categorized yet"]
                
                if industries:
                    st.write("**🏢 Industries Covered:**")
                    for industry in sorted(industries)[:3]:  # Show top 3
                        st.write(f"• {industry}")
                
                # Show relevance
                avg_score = sum(doc["Score"] for doc in last_result["retrieved_docs"]) / len(last_result["retrieved_docs"])
                st.metric("🎯 Avg. Relevance", f"{avg_score:.2f}")
    
    # Display conversation
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💭 Your Career Conversations")
        
        # Show most recent conversations first
        for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):  # Show last 3
            with st.container():
                # User question
                st.markdown(f"""
                <div class="chat-message">
                    <strong>🧑‍🎓 You asked:</strong><br>
                    {chat["query"]}
                </div>
                """, unsafe_allow_html=True)
                
                # AI response
                result = chat["result"]
                if result.get("success"):
                    st.markdown(f"""
                    <div class="response-container">
                        <strong>🤖 Career Advisor:</strong><br><br>
                        {result["final_response"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources in expandable section
                    with st.expander(f"📚 View {len(result['retrieved_docs'])} Professional Sources"):
                        for j, doc in enumerate(result["retrieved_docs"], 1):
                            st.markdown(f"""
                            **Professional {j}:** {doc['Interviewee']}  
                            **Industry:** {doc['Industry Sectors']}  
                            **Relevance:** {doc['Score']:.2f}  
                            **Excerpt:** "{doc['Passage'][:150]}..."
                            """)
                            st.markdown("---")
                else:
                    st.error(f"❌ {result.get('error', 'Unknown error')}")
                
                st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
# Run the Streamlit app