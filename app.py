import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint

st.set_page_config(page_title="\U0001F9E0 Chat with Multi-Docs", layout="wide")
st.title("\U0001F4AC Chat with Your Documents")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize LLM with error handling
try:
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        task="text-generation",
        huggingfacehub_api_token=st.secrets["huggingface"]["token"],
        temperature=0.7,
        max_new_tokens=512
    )
except Exception as e:
    st.error(f"LLM başlatılamadı: {str(e)}")
    st.info("Lütfen Hugging Face API tokenınızı kontrol edin. Tokenınızı https://huggingface.co/settings/tokens adresinden alabilirsiniz.")
    st.stop()

# File uploader
uploaded_files = st.file_uploader(
    "PDF, TXT veya DOCX dosyalarını yükleyin",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []
    for file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
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
            os.unlink(path)  # Clean up temporary file
        except Exception as e:
            st.warning(f"Dosya yüklenirken hata oluştu {file.name}: {str(e)}")

    if all_docs:
        st.success(f"{len(all_docs)} doküman yüklendi.")
        try:
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(all_docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embeddings)
            retriever = db.as_retriever()

            qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

            user_input = st.chat_input("Belgelere dair bir soru sorun...")
            if user_input:
                with st.spinner("Yanıt üretiliyor..."):
                    try:
                        result = qa_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
                        st.session_state.chat_history.append((user_input, result["answer"]))
                    except Exception as e:
                        st.error(f"Soru işlenirken hata: {str(e)}")

            for q, a in st.session_state.chat_history:
                with st.chat_message("user", avatar="\U0001F464"):
                    st.markdown(q)
                with st.chat_message("assistant", avatar="\U0001F916"):
                    st.markdown(a)
        except Exception as e:
            st.error(f"Dokümanlar işlenirken hata: {str(e)}")
    else:
        st.warning("Geçerli doküman yüklenmedi.")
else:
    st.info("Sohbete başlamadan önce en az bir belge yükleyin.")
