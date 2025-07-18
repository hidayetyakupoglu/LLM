import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# API ayarları
os.environ["OPENAI_API_KEY"] = st.secrets["general"]["openrouter_key"]
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="🧠 Chat with Multi-Docs", layout="wide")
st.title("💬 Chat with Your Documents")

# Oturum geçmişi
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader("PDF, TXT veya DOCX dosyalarını yükleyin", type=["pdf", "txt", "docx"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(path)
        else:
            loader = Docx2txtLoader(path)

        docs = loader.load()
        all_docs.extend(docs)

    st.success(f"{len(all_docs)} doküman yüklendi.")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    llm = ChatOpenAI(
        model="mistralai/mistral-7b-instruct",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_API_BASE"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

    user_input = st.chat_input("Belgelere dair bir soru sorun...")
    if user_input:
        with st.spinner("Yanıt üretiliyor..."):
            result = qa_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
            st.session_state.chat_history.append((user_input, result["answer"]))

    # Geçmiş konuşmaları göster
    for q, a in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(a)
else:
    st.info("Sohbete başlamadan önce en az bir belge yükleyin.")
