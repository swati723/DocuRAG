# app.py
# FINAL COMPLETE WORKING VERSION
# SAME FUNCTIONALITY + AESTHETIC UI (SAFE)

import streamlit as st
import pandas as pd
from rag_engine import RAGEngine
from PIL import Image
import io

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Academic RAG Agent",
    page_icon="🎓",
    layout="wide"
)

# ==========================================================
# SAFE UI STYLING (LIGHT + CARDS + SHADOWS)
# ==========================================================

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #eef2ff, #fdf2f8);
}

/* Card */
.card {
    background: #ffffff;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 12px;
}

/* Colored Cards */
.card-blue { background: #eef2ff; }
.card-green { background: #ecfdf5; }
.card-yellow { background: #fffbeb; }
.card-pink { background: #fdf2f8; }

/* Hover */
.card:hover {
    transform: translateY(-2px);
    transition: 0.2s;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
}

/* Buttons */
.stButton>button {
    border-radius: 8px;
    background: linear-gradient(135deg,#6366f1,#ec4899);
    color: white;
    border: none;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

rag_engine = st.session_state.rag_engine

# ==========================================================
# LOAD DOCUMENTS
# ==========================================================

@st.cache_data(ttl=5)
def load_documents():
    try:
        return rag_engine.get_all_documents()
    except:
        return []

docs = load_documents()

# ==========================================================
# METRICS
# ==========================================================

total_docs = len(docs)

total_chunks = sum(
    int(doc.get("chunks", 0))
    for doc in docs
)

all_terms = []

for doc in docs:
    terms = doc.get("key_terms", "")
    if terms:
        for t in terms.split(","):
            t = t.strip()
            if t:
                all_terms.append(t)

unique_terms = len(set(all_terms))

# ==========================================================
# HEADER
# ==========================================================

st.title("🎓 Academic RAG - Policy & Rule Interpretation Agent")
st.caption("📚 Smart Document Analysis for Students & Educators")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f'<div class="card card-pink">📚 Documents<br><b>{total_docs}</b></div>', unsafe_allow_html=True)

with m2:
    st.markdown(f'<div class="card card-blue">📄 Chunks<br><b>{total_chunks}</b></div>', unsafe_allow_html=True)

with m3:
    st.markdown(f'<div class="card card-green">🔑 Key Terms<br><b>{unique_terms}</b></div>', unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.subheader("📚 Upload Documents")

    uploaded_file = st.file_uploader(
        "Upload Course Material",
        type=["pdf", "txt", "csv", "md", "docx"]
    )

    if uploaded_file:

        allowed = ["pdf", "txt", "csv", "md", "docx"]
        ext = uploaded_file.name.split(".")[-1].lower()

        if ext not in allowed:
            st.error("❌ Unsupported file type")
            st.stop()

        if st.button("🚀 Process File", use_container_width=True):

            with st.spinner("Uploading & Processing..."):

                try:
                    file_bytes = uploaded_file.read()

                    chunks = rag_engine.process_upload(
                        file_bytes,
                        uploaded_file.name
                    )

                    if chunks == 0:
                        st.warning("⚠️ File already uploaded.")
                    else:
                        st.success(f"✅ {uploaded_file.name} processed successfully ({chunks} chunks)")

                    st.cache_data.clear()
                    st.rerun()

                except Exception as e:
                    st.error("❌ Error while processing file")
                    st.text(str(e))

    st.divider()

    if st.button("📥 Load Existing Supabase Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Documents",
    "📝 Summaries",
    "🔑 Key Terms",
    "💬 Chat"
])

# ==========================================================
# DOCUMENTS TAB
# ==========================================================

with tab1:

    st.subheader("📚 Document Library")

    if docs:

        rows = []
        for doc in docs:
            rows.append({
                "filename": doc.get("filename", ""),
                "file_type": doc.get("file_type", ""),
                "chunks": doc.get("chunks", 0),
                "uploaded_date": str(doc.get("uploaded_date", ""))[:19]
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("⬇ Download Files")

        for doc in docs:
            url = rag_engine.get_download_url(doc["filename"])
            if url:
                st.markdown(f"""
                <div class="card card-yellow">
                📄 <b>{doc['filename']}</b><br><br>
                <a href="{url}">⬇ Download File</a>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.info("📁 No documents uploaded yet.")

# ==========================================================
# SUMMARY TAB
# ==========================================================

with tab2:

    st.subheader("📝 Document Summaries")

    if docs:
        for doc in docs:
            with st.expander(f"📄 {doc.get('filename','')}"):
                summary = doc.get("summary", "")
                if summary:
                    st.markdown(f"""
                    <div class="card card-blue">
                    {summary}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No summary available.")
    else:
        st.info("No summaries available.")

# ==========================================================
# KEY TERMS TAB
# ==========================================================

with tab3:

    st.subheader("🔑 Key Terms By File")

    if docs:
        for doc in docs:
            with st.expander(f"📄 {doc.get('filename','')}"):
                terms = doc.get("key_terms", "")
                if terms:
                    term_list = [x.strip() for x in terms.split(",") if x.strip()]
                    cols = st.columns(3)
                    for i, term in enumerate(term_list):
                        with cols[i % 3]:
                            st.markdown(f"""
                            <div class="card card-green">
                            {term}
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No key terms available.")
    else:
        st.info("No key terms found.")

# ==========================================================
# CHAT TAB
# ==========================================================

with tab4:

    st.subheader("💬 Ask Questions About Uploaded Documents")

    if not docs:
        st.info("Upload documents first.")

    for msg in st.session_state.chat_history:

        with st.chat_message(msg["role"]):

            st.markdown(msg["content"])

            if msg.get("sources"):

                with st.expander("📎 Relevant Sources"):

                    for i, src in enumerate(msg["sources"], start=1):

                        st.write(f"{i}. 📄 {src['filename']} | Page {src['page']}")

                        if src.get("url"):
                            st.markdown(f"[⬇ Download File]({src['url']})")

# ==========================================================
# CHAT INPUT
# ==========================================================

question = st.chat_input("💡 Ask about your uploaded documents...")

if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.spinner("Searching relevant answer..."):

        answer, sources, ok = rag_engine.answer_question(question)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources if ok else []
    })

    st.rerun()

# ==========================================================
# FOOTER
# ==========================================================

st.caption("✨ Academic RAG System | Powered by AI + Supabase + LangChain")
