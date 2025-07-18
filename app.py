import streamlit as st
import os
import tempfile
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# OpenRouter API bilgilerinizi buraya girin
os.environ["OPENAI_API_KEY"] = st.secrets["openrouter_key"]
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

st.set_page_config(page_title="🧠 Multi-Doc LLM Q&A", layout="wide")
st.title("📄 Multi-Document Q&A with Free LLM (Mistral via OpenRouter)")

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

    st.info("Parçalanıyor ve embedding yapılıyor...")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    llm = ChatOpenAI(
        model="mistralai/mistral-7b-instruct",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_API_BASE"]
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    user_q = st.text_input("Sorunuz:")
    if user_q:
        with st.spinner("Yanıt oluşturuluyor..."):
            ans = qa.run(user_q)
            st.success(ans)
else:
    st.info("Başlamak için en az bir dosya yükleyin.")
