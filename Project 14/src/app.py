import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import streamlit as st

from pdf_loader import load_pdf
from chunker import create_chunks
from embedder import create_embeddings
from vector_store import create_index
from retriever import search
from chatbot import ask_gemini


st.set_page_config(
    page_title="LuminaPDF AI",
    page_icon="🤖",
    layout="wide"
)


# -------------------------
# SESSION STATE
# -------------------------

defaults = {
    "processed": False,
    "chunks": [],
    "index": None,
    "messages": [],
    "pdf_name": "No PDF Uploaded",
    "pages": 0,
    "questions": 0,
    "processing_time": 0,
    "retrieval_time": 0,
    "ai_time": 0,
    "experiment_results": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value



# -------------------------
# STYLE
# -------------------------

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap'
);

html,body,[class*="css"]{
    font-family:'Inter',sans-serif;
}

.stApp{
background:
radial-gradient(circle at top,#4338CA 0%,#111827 45%,#09090B 100%);
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container{
padding-top:2rem;
max-width:1400px;
}

/* Fixed ChatGPT-style chat input */

div[data-testid="stChatInput"] {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 48%;
    z-index: 9999;
}

/* Extra space so messages aren't hidden behind input */

.block-container {
    padding-bottom: 120px;
}

.stButton button{
border-radius:14px;
height:45px;
font-weight:700;
}

/* ChatGPT-style sidebar */

/* ChatGPT-style sidebar */

section[data-testid="stSidebar"] {
    background: #0B1020;
    border-right: 1px solid #252B45;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

/* Keep the sidebar toggle button visible */

button[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 99999 !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

</style>
""", unsafe_allow_html=True)



# -------------------------
# HEADER
# -------------------------

st.markdown("# 🤖 LuminaPDF AI")
st.caption("Next Generation PDF Intelligence")



# -------------------------
# METRICS
# -------------------------

c1, c2, c3, c4, c5, c6 = st.columns(6)

# Get the latest experiment result
latest_result = None

if st.session_state.experiment_results:
    latest_result = st.session_state.experiment_results[-1]


c1.metric(
    "📄 Pages",
    st.session_state.pages
)


c2.metric(
    "✂️ Chunks",
    len(st.session_state.chunks)
)


c3.metric(
    "🧠 Embeddings",
    len(st.session_state.chunks)
)


c4.metric(
    "💬 Questions",
    st.session_state.questions
)


if latest_result:

    c5.metric(
        "⚡ Retrieval",
        f"{latest_result['Retrieval (s)']:.4f}s"
    )

    c6.metric(
        "🤖 AI Response",
        f"{latest_result['AI Response (s)']:.2f}s"
    )

else:

    c5.metric(
        "⚡ Retrieval",
        "0.0000s"
    )

    c6.metric(
        "🤖 AI Response",
        "0.00s"
    )

st.divider()



# -------------------------
# MAIN COLUMNS
# -------------------------


# -------------------------
# CHATGPT-STYLE SIDEBAR
# -------------------------

with st.sidebar:

    st.markdown("# 🤖 LuminaPDF")
    st.caption("PDF Intelligence Workspace")

    st.divider()

    if st.button("＋ New Analysis", use_container_width=True):
        st.session_state.messages = []
        st.session_state.questions = 0
        st.session_state.experiment_results = []

    st.markdown("### 📂 Workspace")

    st.markdown("#### 🧠 Embedding Model")

    embedding_model = st.selectbox(
        "Choose a model",
        [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1"
        ],
        label_visibility="collapsed"
    )

    st.markdown("#### 📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:

        os.makedirs("../data", exist_ok=True)

        pdf_path = "../data/uploaded.pdf"

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.pdf_name = uploaded_file.name

        st.success("PDF Uploaded")

        if st.button("🚀 Process PDF", use_container_width=True):

            with st.spinner("Reading and indexing PDF..."):

                start_time = time.time()

                text = load_pdf(pdf_path)

                chunks = create_chunks(text)

                embeddings = create_embeddings(
                    chunks,
                    embedding_model
                )

                index = create_index(embeddings)

                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.processed = True

                st.session_state.processing_time = (
                    time.time() - start_time
                )

                st.session_state.pages = max(
                    1,
                    text.count("\f") + 1
                )

            st.success("PDF Ready!")

    st.divider()

    st.markdown("### 📄 Current File")

    if st.session_state.pdf_name:
        st.info(st.session_state.pdf_name)
    else:
        st.info("No PDF uploaded")

    st.divider()

    if st.button("🗑 Clear Chat", use_container_width=True):

        st.session_state.messages = []
        st.session_state.questions = 0

        st.rerun()


# -------------------------
# MAIN CONTENT
# -------------------------

middle, right = st.columns([2.8, 1.7])

with middle:

    st.markdown(
        "## 💬 Chat With Your PDF"
    )


    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )



    prompt = st.chat_input(
        "Ask something about your PDF..."
    )

    if prompt:

        if not st.session_state.processed:

            st.warning(
                "Please upload and process a PDF first."
            )

        else:

            # Save user question
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

            # Show user question
            with st.chat_message("user"):

                st.write(prompt)

            # AI response
            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):

                    start_search = time.time()

                    docs = search(
                        st.session_state.index,
                        st.session_state.chunks,
                        prompt,
                        embedding_model
                    )

                    st.session_state.retrieval_time = (
                        time.time() - start_search
                    )

                    start_ai = time.time()

                    answer = ask_gemini(
                        prompt,
                        docs
                    )

                    st.session_state.ai_time = (
                        time.time() - start_ai
                    )

                st.write(answer)

            # Retrieved context
            with st.expander("📚 Retrieved Context"):

                for i, doc in enumerate(docs, 1):

                    st.markdown(
                        f"### Chunk {i}"
                    )

                    st.write(doc)

                    st.divider()

            # Experiment metrics
            total_time = (
                st.session_state.processing_time
                + st.session_state.retrieval_time
                + st.session_state.ai_time
            )

            st.session_state.experiment_results.append(
                {
                    "Model": embedding_model,
                    "Processing (s)": round(
                        st.session_state.processing_time, 3
                    ),
                    "Retrieval (s)": round(
                        st.session_state.retrieval_time, 4
                    ),
                    "AI Response (s)": round(
                        st.session_state.ai_time, 3
                    ),
                    "Total (s)": round(
                        total_time, 3
                    )
                }
            )

            # Save AI answer
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            # Increase question counter
            st.session_state.questions += 1
      

# Refresh dashboard metrics



# -------------------------
# RIGHT PANEL
# -------------------------

# -------------------------
# RIGHT PANEL
# -------------------------

with right:

    st.markdown("## 🤖 AI Status")

    if st.session_state.processed:

        st.success("PDF Connected")

        st.write(
            "Ready to answer questions."
        )

        st.divider()

        # -------------------------
        # PERFORMANCE
        # -------------------------

        st.markdown("### 📈 Performance")

        if st.session_state.experiment_results:

            df = pd.DataFrame(
                st.session_state.experiment_results
            )

            # Processing Time
            st.markdown("#### ⏱ Processing Time")

            fig1, ax1 = plt.subplots(
                figsize=(6, 2.5)
            )

            ax1.bar(
                df["Model"],
                df["Processing (s)"]
            )

            ax1.set_ylabel("Seconds")
            ax1.tick_params(
                axis="x",
                labelrotation=25,
                labelsize=7
            )

            plt.tight_layout()

            st.pyplot(
                fig1,
                use_container_width=True
            )

            # Retrieval Time
            st.markdown("#### ⚡ Retrieval Time")

            fig2, ax2 = plt.subplots(
                figsize=(6, 2.5)
            )

            ax2.bar(
                df["Model"],
                df["Retrieval (s)"]
            )

            ax2.set_ylabel("Seconds")
            ax2.tick_params(
                axis="x",
                labelrotation=25,
                labelsize=7
            )

            plt.tight_layout()

            st.pyplot(
                fig2,
                use_container_width=True
            )

            # AI Response Time
            st.markdown("#### 🤖 AI Response")

            fig3, ax3 = plt.subplots(
                figsize=(6, 2.5)
            )

            ax3.bar(
                df["Model"],
                df["AI Response (s)"]
            )

            ax3.set_ylabel("Seconds")
            ax3.tick_params(
                axis="x",
                labelrotation=25,
                labelsize=7
            )

            plt.tight_layout()

            st.pyplot(
                fig3,
                use_container_width=True
            )

        else:

            st.info(
                "Run an experiment to see performance data."
            )

    else:

        st.warning("Waiting for PDF")


st.divider()

st.caption(
    "LuminaPDF AI • Powered by Groq"
)
st.divider()

st.markdown("## 📊 Embedding Model Comparison")

if st.session_state.experiment_results:

    st.dataframe(
        st.session_state.experiment_results,
        use_container_width=True
    )

else:

    st.info("Run experiments to compare embedding models.")

 