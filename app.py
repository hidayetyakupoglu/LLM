import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

st.set_page_config(page_title="\U0001F9E0 Chat with Multi-Docs", layout="wide")
st.title("\U0001F4AC Chat with Your Documents")

# LLM nesnesi tek yerde, token parametresi ile tanımlanıyor
from langchain_community.llms import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",  # Daha büyük ve erişilebilir bir model
    huggingfacehub_api_token="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",
    temperature=0.7,
    max_length=512
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    "PDF, TXT veya DOCX dosyalarını yükleyin",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

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

    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

    user_input = st.chat_input("Belgelere dair bir soru sorun...")
    if user_input:
        with st.spinner("Yanıt üretiliyor..."):
            result = qa_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
            st.session_state.chat_history.append((user_input, result["answer"]))

    for q, a in st.session_state.chat_history:
        with st.chat_message("user", avatar="\U0001F464"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="\U0001F916"):
            st.markdown(a)

else:
    st.info("Sohbete başlamadan önce en az bir belge yükleyin.")
