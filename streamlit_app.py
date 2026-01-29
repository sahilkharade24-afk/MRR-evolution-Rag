import streamlit as st
from rag_chain import create_rag_chain
from evaluate_mmr import evaluate_mmr_tuning, generate_rank_chart
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="RAG-MMR Evaluation Chat",
    page_icon="🤖",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Custom CSS for Background and Aesthetics
st.markdown("""
    <style>
    /* Premium Mesh Gradient Background */
    .stApp {
        background-color: #0d1117;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%), 
            radial-gradient(at 0% 100%, hsla(321,50%,30%,1) 0, transparent 50%), 
            radial-gradient(at 100% 100%, hsla(11,46%,30%,1) 0, transparent 50%);
        background-attachment: fixed;
    }

    /* Force High Contrast Text (Generic) */
    .stApp, .stMarkdown, p, span, label, .stMetric label {
        color: #ffffff !important;
    }

    /* Glassmorphism for Containers */
    div.stChatMessage, div.stExpander, div.stMetric, .stDataFrame, .stTable {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        margin-bottom: 20px !important;
    }

    /* Specific Table and Dataframe Styling for visibility */
    .stTable table {
        border-collapse: collapse !important;
        width: 100% !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .stTable th {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #00ffcc !important;
        font-weight: bold !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 10px !important;
    }
    
    .stTable td {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 8px !important;
        color: #ffffff !important;
    }

    /* Fix Expander Opaque Background */
    .stExpander, .stExpander > details, .stExpander > details > summary {
        background-color: transparent !important;
    }
    
    .stExpander > details > summary:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }

    /* Force High Contrast Text (Generic) */
    .stApp, .stMarkdown, p, span, label, .stMetric label, .stExpander summary p {
        color: #ffffff !important;
    }

    /* Specific Metric Value Styling */
    div[data-testid="stMetricValue"] > div {
        color: #00ffcc !important; 
        font-weight: 800 !important;
    }

    /* Tab Styling */
    button[data-baseweb="tab"] p {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    button[aria-selected="true"] p {
        color: #00ffcc !important;
    }

    /* Button Styling (Prevent white-out) */
    .stButton > button {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: rgba(0, 255, 204, 0.2) !important;
        border-color: #00ffcc !important;
    }

    /* Header Styling */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.6) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Ensure Expander content is also readable */
    .stExpander div[data-testid="stExpanderDetails"] {
        color: white !important;
        background-color: transparent !important;
    }

    /* Input Styling */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    with st.spinner("Initializing RAG chain..."):
        st.session_state.rag_chain = create_rag_chain(return_context=True)

if "live_feedback" not in st.session_state:
    st.session_state.live_feedback = []

# Sidebar for Auto-Attribution
st.sidebar.title("Configuration")
auto_attribute = st.sidebar.checkbox("Enable Auto-Attribution", value=False)

if auto_attribute and "embeddings_model" not in st.session_state:
    with st.spinner("Loading embedding model for auto-attribution..."):
        st.session_state.embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def render_message(msg_idx, message):
    """Helper to render a chat message with context and feedback."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "context" in message:
            with st.expander("Show Retrieved Context", expanded=False):
                for doc_idx, doc in enumerate(message["context"]):
                    st.markdown(f"**Source {doc_idx+1}** (chunk_id: {doc.metadata.get('chunk_id')})")
                    st.text_area(f"Content {doc_idx+1}", value=doc.page_content, height=150, disabled=True, key=f"txt_{msg_idx}_{doc_idx}")
                    
                    # Feedback Checkbox
                    checkbox_key = f"fb_{msg_idx}_{doc_idx}"
                    if st.checkbox(f"Mark Source {doc_idx+1} as Relevant", key=checkbox_key):
                        # Find if this specific rank is already recorded for this message
                        existing = next((f for f in st.session_state.live_feedback if f['msg_idx'] == msg_idx and f['relevant_rank'] == doc_idx + 1), None)
                        if not existing:
                            query_text = st.session_state.messages[msg_idx-1]['content'] if msg_idx > 0 else "Unknown"
                            st.session_state.live_feedback.append({
                                'msg_idx': msg_idx,
                                'query': query_text,
                                'relevant_rank': doc_idx + 1
                            })
                            st.toast(f"Source {doc_idx+1} marked as relevant!")

st.title("🤖 RAG-MMR Evaluation Suite")

# Tabs
tab1, tab2 = st.tabs(["💬 Interactive Chat", "📊 Evaluation Dashboard"])

with tab1:
    st.markdown("Ask questions and mark the retrieved sources as 'Relevant' to update live stats.")

    # Render History
    for i, msg in enumerate(st.session_state.messages):
        render_message(i, msg)

    # Chat Input
    if prompt := st.chat_input("What would you like to know?"):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun() # Rerun to show user message immediately and move to assistant

with tab2:
    st.header("Retriever Performance Analytics")
    
    # 1. Automated MRR Grid Search
    with st.expander("🛠️ Automated Parameter Tuning", expanded=False):
        if st.button("Run Full Evaluation Grid Search"):
            with st.spinner("Running evaluation..."):
                results = evaluate_mmr_tuning(generate_chart=False)
                if results:
                    st.success("Evaluation Complete!")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Best MRR", f"{results['best_mrr']:.4f}")
                    m2.metric("Best Recall", f"{results['best_recall']:.4f}")
                    m3.metric("Best Precision", f"{results['best_precision']:.4f}")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Best Parameters:**", results['best_params'])
                        st.pyplot(generate_rank_chart(results['best_ranks']))
                    with c2:
                        st.dataframe(pd.DataFrame(results['all_results']).drop(columns=['ranks']))
                        # Star Rating
                        num_stars = max(1, min(5, int(round(results['best_mrr'] * 5))))
                        stars = "★" * num_stars + "☆" * (5 - num_stars)
                        st.markdown(f"### Quality: <span style='color:gold; font-size:30px;'>{stars}</span>", unsafe_allow_html=True)

    st.write("---")
    
    # 2. Live Interaction Analytics
    st.header("📈 Live Interaction Analytics")
    if not st.session_state.live_feedback:
        st.info("No live feedback yet. Go to Chat and mark sources as relevant!")
    else:
        df_live = pd.DataFrame(st.session_state.live_feedback)
        total_queries = df_live['msg_idx'].nunique()
        total_relevant = len(df_live)
        
        # Calculate Live MRR (best rank per query)
        best_ranks = df_live.groupby('msg_idx')['relevant_rank'].min()
        live_mrr = sum(1/r for r in best_ranks) / total_queries
        
        # Calculate Live Precision (average relevant docs out of 5 retrieved)
        live_precision = total_relevant / (total_queries * 5)
        
        # Calculate Live Recall (fraction of queries with at least one match - usually 1.0 when active)
        live_recall = total_queries / total_queries # Placeholder for more complex recall if needed
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Live Queries", total_queries)
        m2.metric("Live MRR", f"{live_mrr:.4f}")
        m3.metric("Live Precision@5", f"{live_precision:.2f}")
        m4.metric("Top-1 Accuracy", f"{(len(df_live[df_live['relevant_rank']==1])/total_queries)*100:.1f}%")
        
        st.subheader("Feedback Log")
        st.table(df_live[['query', 'relevant_rank']])

# Handle Assistant Logic outside the input block for consistency
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke(st.session_state.messages[-1]["content"])
            
            # Robust handling of different chain output formats
            if isinstance(response, dict):
                ans_text = response.get('answer', str(response))
                ctx_docs = response.get('context', [])
            else:
                ans_text = str(response)
                ctx_docs = []

            st.session_state.messages.append({
                "role": "assistant",
                "content": ans_text,
                "context": ctx_docs
            })

            # Auto-Attribution Logic
            if auto_attribute and "embeddings_model" in st.session_state and ctx_docs:
                try:
                    # 1. Embed Answer
                    ans_emb = st.session_state.embeddings_model.embed_query(ans_text)
                    
                    # 2. Embed all context chunks
                    ctx_embs = st.session_state.embeddings_model.embed_documents([d.page_content for d in ctx_docs])
                    
                    # 3. Calculate Similarities
                    sims = cosine_similarity([ans_emb], ctx_embs)[0]
                    
                    # 4. Find Best Match
                    best_idx = np.argmax(sims)
                    best_score = sims[best_idx]
                    
                    # 5. Auto-Mark
                    # Only mark if similarity is reasonably high (e.g., > 0.3)
                    if best_score > 0.3:
                         msg_idx = len(st.session_state.messages) - 1 # Current message index
                         st.session_state.live_feedback.append({
                            'msg_idx': msg_idx,
                            'query': st.session_state.messages[-2]['content'], # User query
                            'relevant_rank': int(best_idx) + 1
                         })
                         st.toast(f"Auto-Attributed to Source {best_idx+1} (Score: {best_score:.2f})")
                except Exception as e:
                    print(f"Auto-attribution error: {e}")
    st.rerun()
