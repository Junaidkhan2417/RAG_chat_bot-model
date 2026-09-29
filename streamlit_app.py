import streamlit as st
import time
from src.graph import build_rag_graph
from src.config import (
    PINECONE_INDEX_NAME,
    OPENAI_MODEL,
    OPENAI_EMBEDDING_MODEL,
    TOP_K,
    has_live_credentials,
)

# Page configuration
st.set_page_config(
    page_title="Agentic AI RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #713F12;
    }
    .chunk-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize LangGraph workflow in session state
@st.cache_resource
def load_graph():
    return build_rag_graph(index_name=PINECONE_INDEX_NAME)

graph = load_graph()

# Sidebar: System configuration and benchmark queries
with st.sidebar:
    st.title("⚙️ RAG System Config")
    live_status = has_live_credentials()
    if live_status:
        st.success("🟢 Live Cloud Mode: Connected to Pinecone & OpenAI")
    else:
        st.warning("🟡 Demonstration Mode: Using local grounded chunks")

    st.markdown("---")
    st.markdown("### 📋 Model Parameters")
    st.text(f"LLM: {OPENAI_MODEL}")
    st.text(f"Embeddings: {OPENAI_EMBEDDING_MODEL}")
    st.text(f"Pinecone Index: {PINECONE_INDEX_NAME}")
    st.text(f"Retriever Top-k: {TOP_K}")

    st.markdown("---")
    st.markdown("### 🧪 Benchmark Queries")
    sample_queries = [
        "What is Agentic AI according to the eBook?",
        "How do AI agents differ from traditional automation systems?",
        "What are the core components of an Agentic Architecture?",
        "What role does memory play in Agentic AI workflows?",
        "Who won the 2022 FIFA World Cup?",
    ]

    selected_query = None
    for q in sample_queries:
        if st.button(f"📌 {q[:38]}...", help=q, use_container_width=True):
            selected_query = q

    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main area
st.markdown('<div class="main-title">🤖 Agentic AI RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Stateful RAG chatbot strictly grounded in the official <i>Agentic AI eBook</i> using <b>LangGraph</b>, <b>Pinecone</b>, and <b>FastAPI</b>.</div>',
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context" in msg and msg["context"]:
            with st.expander(f"🔍 Retrieved Chunks ({len(msg['context'])}) & Confidence Score: {msg.get('score', 0.0):.2f}"):
                st.progress(float(msg.get("score", 0.0)))
                for idx, c in enumerate(msg["context"]):
                    st.markdown(f"**Chunk {idx + 1}:**")
                    st.markdown(f'<div class="chunk-box">{c}</div>', unsafe_allow_html=True)

# User input handling
user_input = st.chat_input("Ask a question about the Agentic AI eBook...") or selected_query

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process query through LangGraph workflow
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from Pinecone & generating grounded answer..."):
            start_time = time.time()
            initial_state = {
                "question": user_input,
                "context": [],
                "answer": "",
                "score": 0.0,
            }
            res = graph.invoke(initial_state)
            latency = time.time() - start_time

            answer = res.get("answer", "I cannot answer based on the provided document.")
            context = res.get("context", [])
            score = float(res.get("score", 0.0))

            st.markdown(answer)

            # Context & confidence panel
            badge_class = "badge-success" if score > 0.5 else "badge-warning"
            st.markdown(
                f'<span class="metric-badge {badge_class}">Confidence Score: {score:.2f} | Latency: {latency:.2f}s</span>',
                unsafe_allow_html=True,
            )

            if context:
                with st.expander(f"📄 Inspect Retrieved Context Chunks ({len(context)})"):
                    for idx, c in enumerate(context):
                        st.markdown(f"**Source Chunk {idx + 1}** ({len(c)} characters):")
                        st.markdown(f'<div class="chunk-box">{c}</div>', unsafe_allow_html=True)
            else:
                st.info("ℹ️ No relevant context chunks found in document for this query.")

    # Save assistant message to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": context,
        "score": score,
    })
